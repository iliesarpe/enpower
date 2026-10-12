"""Downloads every dataset of the experiments, and the CLIP weights, into data/.

    python download_data.py                    # everything (about 3 GB)
    python download_data.py --only tabular     # the six OpenML datasets
    python download_data.py --only clip        # the three image datasets and CLIP ViT-B/32

Anything already present is skipped, so the script can be rerun after an interruption.
After it, no experiment needs the network.

  data/openml/                      the OpenML cache of scikit-learn (fetch_openml); the training
                                    scripts read it through SCIKIT_LEARN_DATA, which reproduce.sh sets
  data/cifar-100-python/            CIFAR-100, test split (torchvision)
  data/oxford-iiit-pet/             Oxford-IIIT Pet, test split (torchvision)
  data/imagenet-a/                  ImageNet-A (Hendrycks et al.); if its official link is down,
                                    set IMAGENET_A_URL to a mirror or extract imagenet-a.tar here
  data/imagenet_class_index.json    names of the ImageNet classes, for ImageNet-A
  data/clip/                        the weights of CLIP ViT-B/32
"""
import argparse
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# The tabular datasets of the paper, by OpenML id
OPENML = {31: "credit-g", 1461: "bank-marketing", 1464: "blood-transfusion-service-center",
          1468: "cnae-9", 40945: "titanic", 150: "Covertype"}


def tabular(data_root, ids):
    from sklearn.datasets import fetch_openml

    home = os.environ.get("SCIKIT_LEARN_DATA", os.path.join(data_root, "openml"))
    for i in ids:
        print("OpenML %d (%s) -> %s" % (i, OPENML.get(i, "?"), home), flush=True)
        fetch_openml(data_id=i, as_frame=True, parser="auto", data_home=home)


def clip(data_root):
    sys.path.insert(0, HERE)
    import clip as openai_clip
    import torchvision
    from clip_inference import IMAGENET_CLASS_INDEX_URL, MODEL_NAME, fetch_imagenet_a

    print("CIFAR-100 -> %s" % data_root, flush=True)
    torchvision.datasets.CIFAR100(root=data_root, train=False, download=True)
    print("Oxford-IIIT Pet -> %s" % data_root, flush=True)
    torchvision.datasets.OxfordIIITPet(root=data_root, split="test", download=True)

    print("ImageNet-A -> %s" % data_root, flush=True)
    fetch_imagenet_a(data_root)
    index = os.path.join(data_root, "imagenet_class_index.json")
    if not os.path.exists(index):
        print("ImageNet class index -> %s" % index, flush=True)
        urllib.request.urlretrieve(IMAGENET_CLASS_INDEX_URL, index)

    print("CLIP %s -> %s" % (MODEL_NAME, os.path.join(data_root, "clip")), flush=True)
    openai_clip.load(MODEL_NAME, device="cpu", download_root=os.path.join(data_root, "clip"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["tabular", "clip"], default=None)
    parser.add_argument("--openml-ids", nargs="*", type=int, default=list(OPENML),
                        help="the OpenML datasets to fetch (default: the six of the paper)")
    parser.add_argument("--data_root", default=os.path.join(REPO, "data"))
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)
    if args.only in (None, "tabular"):
        tabular(args.data_root, args.openml_ids)
    if args.only in (None, "clip"):
        clip(args.data_root)
    print("done")


if __name__ == "__main__":
    main()
