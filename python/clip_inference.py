"""CLIP prompt ensembling: the votes of 22 prompt templates of CLIP ViT-B/32 on an image dataset.

With --votes-only it stores the vote of every prompt on every image, in dataset order, to
<votes_dir>/<dataset>_ViT-B-32_votes.npz; the five random splits and the indices are then computed by
`cpp/tools/enpower.py clip run`. Without it, it also takes the split of seed 100, computes the
indices of the Python reference implementation on D_val, and stores them in
<output_dir>/<dataset>_ViT-B-32_results.json, which clip_prompt_table.py turns into the per-prompt tables.

The datasets and the CLIP weights are read from --data_root (download_data.py fetches them).
"""
import os
import json
import argparse
import torch
import clip
import numpy as np
import torchvision

from scipy.stats import mode
from torchvision.datasets.utils import download_and_extract_archive
import urllib.request


from banzhaf_multiclass import compute_banzhaf_multiclass
from shapley_multiclass import compute_shapley_multiclass

MODEL_NAME = "ViT-B/32"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(REPO, "data")
IMAGENET_CLASS_INDEX_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"
# The official location of ImageNet-A (Hendrycks et al.); set IMAGENET_A_URL to a mirror if it
# is unavailable, or extract imagenet-a.tar into <data_root>/imagenet-a/ manually
IMAGENET_A_URL = os.environ.get("IMAGENET_A_URL", "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar")


def fetch_imagenet_a(data_root):
    """Downloads and extracts ImageNet-A into <data_root>/imagenet-a/, unless it is already present."""
    if os.path.exists(os.path.join(data_root, 'imagenet-a')):
        return
    print("\nImageNet-A not found. Downloading and extracting (this may take a few minutes)...")
    try:
        download_and_extract_archive(IMAGENET_A_URL, download_root=data_root, extract_root=data_root)
    except Exception as e:
        raise SystemExit("ImageNet-A could not be downloaded from %s (%s).\n"
                         "Set IMAGENET_A_URL to a mirror of imagenet-a.tar, or extract it by hand so that "
                         "%s holds its 200 class folders." % (IMAGENET_A_URL, e, os.path.join(data_root, 'imagenet-a')))


def clip_votes(dataset_name, device, data_root=DATA_ROOT):
    """Steps 1-4: the vote of every prompt template on every image of the dataset.

    Returns (prediction_matrix, true_labels, prompt_templates, classes), with one
    row per image in the order of the torchvision dataset. Everything after this
    step is a function of these votes and of the split seed only."""
    # ==========================================
    # 1. model and prompt templates
    # ==========================================
    model_name = MODEL_NAME
    model, preprocess = clip.load(model_name, device=device,
                                  download_root=os.path.join(data_root, 'clip'))

    prompt_templates = [
        "A photo of a {}.",
        "A blurry photo of a {}.",
        "A pixelated photo of a {}.",
        "A heavily compressed JPEG photo of a {}.",
        "A poor quality, grainy photo of a {}.",
        "A high-resolution, crisp photo of a {}.",
        "A sketch of a {}.",
        "A macro shot of a {}.",
        "A painting of a {}.",
        "A black and white photo of a {}.",
        "A photo focusing closely on the details of a {}.",
        "A hand-drawn illustration of a {}.",
        "A photo focusing on the shape of a {}.",
        "A photo focusing on the texture of a {}.",
        "An unusual, hard to recognize photo of a {}.",
        "A photo of a {}, cluttered background.",
        "A cropped photo of a {}.",
        "A photo of a {} in its natural environment.",
        "A photo of a {}, with a watermark.",
        "A photo of a {}, with text over it.",
        "A close-up photo of a {}.",
        "A zoomed-out photo of a {}."
    ]
    num_models = len(prompt_templates)

    # ==========================================
    # 2. dataset
    # ==========================================
    if dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
        classes = dataset.classes
    elif dataset_name == 'pets':
        dataset = torchvision.datasets.OxfordIIITPet(root=data_root, split='test', download=True)
        classes = [c.replace('_', ' ') for c in dataset.classes]
    elif dataset_name == 'imagenet-a':
        imagenet_a_dir = os.path.join(data_root, 'imagenet-a')

        fetch_imagenet_a(data_root)
        dataset = torchvision.datasets.ImageFolder(root=imagenet_a_dir)
        
        index_path = os.path.join(data_root, 'imagenet_class_index.json')
        if os.path.exists(index_path):
            with open(index_path) as f:
                class_idx = json.load(f)
        else:
            response = urllib.request.urlopen(IMAGENET_CLASS_INDEX_URL)
            class_idx = json.loads(response.read())
        
        wnid_to_name = {v[0]: v[1].replace('_', ' ') for k, v in class_idx.items()}
        classes = [wnid_to_name[wnid] for wnid in dataset.classes]
    
    num_samples = len(dataset)
    print(f"Loaded {dataset_name} with {num_samples} images and {len(classes)} classes.")

    # ==========================================
    # 3. text embeddings of the prompts
    # ==========================================
    ensemble_text_features = []
    with torch.no_grad():
        for template in prompt_templates:
            texts = [template.format(c) for c in classes]
            text_tokens = clip.tokenize(texts).to(device)
            text_features = model.encode_text(text_tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            ensemble_text_features.append(text_features)

    # ==========================================
    # 4. inference
    # ==========================================
    prediction_matrix = np.zeros((num_samples, num_models), dtype=int)
    true_labels = np.zeros(num_samples, dtype=int)

    with torch.no_grad():
        for i in range(num_samples):
            image_obj, true_label = dataset[i]
            image = preprocess(image_obj).unsqueeze(0).to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            for m_idx, text_features in enumerate(ensemble_text_features):
                similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                prediction_matrix[i, m_idx] = similarities.argmax(dim=-1).item()
                
            true_labels[i] = true_label

    return prediction_matrix, true_labels, prompt_templates, classes


def votes_path(output_dir, dataset_name):
    """The cache of the votes of clip_votes, in dataset order."""
    return os.path.join(output_dir, f"{dataset_name}_{MODEL_NAME.replace('/', '-')}_votes.npz")


def main():
    parser = argparse.ArgumentParser(description="Votes of the CLIP prompt ensemble on an image dataset")
    parser.add_argument('--dataset', type=str, choices=['cifar100', 'pets', 'imagenet-a'], required=True, 
                        help="Which dataset to run the experiment on.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Hardware device to use (cuda or cpu).")
    parser.add_argument('--output_dir', type=str, default=os.path.join(REPO, "outputs"),
                        help="Directory to save JSON and NPZ files.")
    parser.add_argument('--votes-only', action='store_true',
                        help="only cache the votes of every prompt on every image, in dataset "
                             "order, to <votes_dir>/<dataset>_<model>_votes.npz, and stop; the "
                             "splits and the indices are computed by cpp/tools/enpower.py clip")
    parser.add_argument('--votes_dir', type=str, default=os.path.join(REPO, "outputs", "clip"),
                        help="Directory of the vote cache of --votes-only.")
    parser.add_argument('--data_root', type=str, default=DATA_ROOT,
                        help="Where the image datasets and the CLIP weights are stored.")
    args = parser.parse_args()

    print(f"Using device: {args.device}")
    model_name = MODEL_NAME
    safe_model_name = model_name.replace("/", "-") 
    prediction_matrix, true_labels, prompt_templates, classes = clip_votes(
        args.dataset, args.device, args.data_root)
    num_samples, num_models = prediction_matrix.shape

    if args.votes_only:
        os.makedirs(args.votes_dir, exist_ok=True)
        path = votes_path(args.votes_dir, args.dataset)
        np.savez_compressed(path, preds=prediction_matrix, labels=true_labels,
                            prompts=np.array(prompt_templates), classes=np.array(classes),
                            model=np.array(model_name))
        print(f"Votes cached to {path}")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # ==========================================
    # 5. 50/50 split into D_val and D_test
    # ==========================================
    print("Splitting dataset 50/50 for Selection vs. Evaluation...")
    
    # random permutation of the indices, seeded
    np.random.seed(100)
    shuffled_indices = np.random.permutation(num_samples)
    
    # size of D_val
    split_idx = int(num_samples * 0.5)
    
    # indices of D_val and D_test
    val_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]
    
    # split of the votes and of the labels
    preds_val = prediction_matrix[val_indices]
    labels_val = true_labels[val_indices]
    preds_test = prediction_matrix[test_indices]
    labels_test = true_labels[test_indices]

    # store the split
    npz_path = os.path.join(args.output_dir, f"{args.dataset}_{safe_model_name}_predictions.npz")
    np.savez_compressed(npz_path, 
                        preds_val=preds_val, labels_val=labels_val,
                        preds_test=preds_test, labels_test=labels_test)
    print(f"Predictions and splits cached to {npz_path}")

    # ==========================================
    # 6. metrics and power indices (on D_val)
    # ==========================================
    print(f"Computing metrics on 50% Selection Split ({len(labels_val)} samples)...")
    
    # accuracy of each prompt on D_val
    accuracies = [(preds_val[:, m] == labels_val).mean() for m in range(num_models)]
    
    # accuracy of the plurality-voting ensemble on D_test, for logging
    plurality_votes, _ = mode(preds_test, axis=1, keepdims=False)
    ensemble_accuracy_test = (plurality_votes == labels_test).mean()

    # power indices, computed on D_val only
    print("Computing Banzhaf...")
    banzhaf_vals = compute_banzhaf_multiclass(preds_val, labels_val)
    
    print("Computing Shapley...")
    shapley_vals = compute_shapley_multiclass(preds_val, labels_val, len(classes))

    # ==========================================
    # 7. JSON output
    # ==========================================
    results_cache = {
        "metadata": {
            "dataset": args.dataset,
            "model": model_name,
            "total_samples": num_samples,
            "val_samples": len(labels_val),
            "test_samples": len(labels_test),
            "num_classes": len(classes),
            "ensemble_accuracy_test": float(ensemble_accuracy_test),
            "npz_file_path": os.path.basename(npz_path)
        },
        "sentences": prompt_templates,
        "accuracy": [float(a) for a in accuracies],
        "banzhaf": [float(b) for b in banzhaf_vals],
        "shapley": [float(s) for s in shapley_vals]
    }

    json_path = os.path.join(args.output_dir, f"{args.dataset}_{safe_model_name}_results.json")
    with open(json_path, "w") as f:
        json.dump(results_cache, f, indent=4)
    
    print(f"Experiment complete! Results cached to {json_path}")

if __name__ == "__main__":
    main()
