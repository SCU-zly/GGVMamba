## Features

- Competitive performance in BAA
- Using Vision Mamba and Graph structure

## Future Plans

- Release a fully polished version post paper acceptance
- Additional documentation and user guides

Thank you for your interest and understanding!

---

Feel free to reach out if you have any questions or need further information.

## Data preparation and annotation availability

Verified on 2026-10-20. Dataset access and the released implementation are separate: this repository does not include bone-age CSVs, ROI JSONs, images, or the original experiment split manifests. Local filenames below describe the loader contract, not downloadable assets or verified copies of the original releases.

### Dataset identity and original sources

| Dataset | Released GGVMamba support | Original access route and verification |
| --- | --- | --- |
| RSNA Pediatric Bone Age Challenge (2017) | `dataset='RSNA'`, train and validation branches | [RSNA official challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017) lists training images, training annotations and validation images. Its training-annotation ZIP endpoint returned HTTP 403 during this check; a working archive download could not be verified. |
| Radiological Hand Pose Estimation (RHPE) | `dataset='RHPE'`, train and validation branches | Dataset creators' [BAAR page](https://bcv-uniandes.github.io/baar-wp/) links images, annotations, terms and evaluation. The linked [RHPE annotation archive](https://drive.google.com/file/d/1x4E_xbcqm1UYUFQBBzUfHkVMkr-A72WO/view) resolves to a Drive preview/sign-in page; archive contents and anonymous download were not verified. |
| Digital Hand Atlas (DHA) | No released dataset adapter or split definition | Original lab's [Digital Hand Atlas portal](https://ipilab.usc.edu/research/baaweb/) is an access route, not a verified direct annotation download. It returned HTTP 502 during this check. The indexed page describes an email-based download request; actual delivery, annotation packaging and redistribution permission remain unverified. |

**DHA and RHPE are distinct datasets.** The old DHA comments did not implement DHA loading. Selecting DHA must not silently select the RHPE paths or preprocessing. No GGVMamba-specific DHA CSV, conversion script, or confirmed release date is available in this repository.

BAAR also links an [RSNA annotation archive](https://drive.google.com/file/d/1OBUTusKtx6k5BUEBhqav_DJbFIQ0zxlB/view). The creators describe RSNA/RHPE training and validation bone-age, box and keypoint resources in [BoNet](https://github.com/BCV-Uniandes/Bonet). The Drive preview was reachable, but the archive contents were not inspected; this is not evidence that challenge test labels are included.

### Splits and test labels

RSNA's original challenge has 12,611 training, 1,425 validation and 200 test images ([official dataset description, linked from RSNA](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017)). `ANN_PATH_VAL.csv` is only the local file used by `is_train=False`; its filename does not establish whether its rows match the official validation set, a custom holdout, or any test set. No sample-ID manifest is committed to establish that correspondence. Do not report results on this branch as official 200-image test results without verifying the IDs and labels.

No working official public download for the **200-image RSNA test ground truth** was verified. The current RSNA landing page does not list a test-label download. This does not establish that labels can never be obtained: consult the dataset organizers for current access. BAAR links an evaluation service; its operational status and supported submissions were not validated here. Do not substitute validation labels or model predictions for test ground truth.

For RHPE, preserve the split IDs supplied with your authorized source. BAAR currently lists 5,487/713/79 train/validation/test images, while its original publication description cites 6,288 total images; record the actual release version and IDs rather than assuming these totals are interchangeable. The released GGVMamba builder has no test branch. RHPE test evaluation should follow the creators' instructions; do not assume test bone-age labels are bundled with the public training/validation resources.

### CSV contract of `BAADataset`

CSV files must have a header. The loader reads **column positions**, not header names. Remove saved DataFrame index columns (`Unnamed: 0`); reorder source columns explicitly before use.

| Position | Meaning | RSNA | RHPE |
| --- | --- | --- | --- |
| 0 | Image ID, without `.png` | `<id>.png` | ID padded to five characters, `<id:05>.png` |
| 1 | Numeric sex indicator | Required | Required |
| 2 | Bone age | Required | Required |
| 3 | Chronological age | Ignored; returned as zero | Required |

Use bone age and chronological age consistently in **months**. The loader performs no year-to-month conversion. Check the original metadata's sex coding and use one consistent numeric mapping throughout training/evaluation; the original GGVMamba mapping cannot be established from the absent CSVs. Do not multiply string values such as `"False"` and treat them as booleans.

For example, a prepared RSNA header is `id,male,boneage`. If your source uses `id,boneage,male`, reorder it before loading. Preserve IDs and genuine labels; the loader requires labels even for evaluation and does not support unlabeled inference. Missing labels must not be replaced with fabricated zeros.

### ROI JSON and image preprocessing

For RHPE, each image directory may have a corresponding COCO-style JSON with `images` entries (`id`, `file_name`) and `annotations` entries (`image_id`, `bbox`, `keypoints`). The filename must match the padded PNG name. `bbox` is `[x, y, width, height]` in pixels. The current crop function reads `keypoints` but only uses the box; it takes the first matching annotation. Ensure it is the intended hand box, particularly for bilateral radiographs. JSON resources are separate from image-level bone-age labels.

Passing `json_file=None` explicitly to `BAADataset` selects the legacy left-half image crop for RHPE. That pixel-side heuristic is not guaranteed to select the anatomical left hand. The width/height calculation has been corrected for non-square images, so this fallback can differ from older runs. RSNA uses whole RGB images and does not require ROI JSON in this loader.

### Portable local paths

Set the root before running the existing entry point:

```bash
export GGVMAMBA_DATA_ROOT=/absolute/path/to/BoneData
```

The builder expects these paths below that root:

| Dataset | Images | CSV | ROI JSON |
| --- | --- | --- | --- |
| RSNA train | `RSNA/DATA_TRAIN/` | `RSNA/ANN_PATH_TRAIN.csv` | Not needed |
| RSNA validation | `RSNA/DATA_VAL/` | `RSNA/ANN_PATH_VAL.csv` | Not needed |
| RHPE train | `RHPE/RHPE_TRAIN/` | `RHPE/RHPE_ANNOTATIONS/RHPE_Boneage_train.csv` | `RHPE/RHPE_ANNOTATIONS/RHPE_anatomical_ROIs_train.json` |
| RHPE validation | `RHPE/RHPE_VAL/` | `RHPE/RHPE_ANNOTATIONS/RHPE_Boneage_val.csv` | `RHPE/RHPE_ANNOTATIONS/RHPE_anatomical_ROIs_val.json` |

For different filenames, instantiate the loader directly (replace paths with your authorized files):

```python
from dataset_boneAge import BAADataset

dataset = BAADataset(
    img_dir=["/absolute/path/to/images"],
    ann_file=["/absolute/path/to/prepared_labels.csv"],
    input_size=512,
    dataset="RSNA",
    json_file=None,
)
image, sex, chronological_age, bone_age = dataset[0]
```

Returned values are float tensors: image, sex (length 1), chronological age (length 1), scalar bone age. The default transform resizes and scales to `[0, 1]`; custom transforms in `main.py` use different normalization. Record the transform as part of an experiment. Without the root variable, legacy server paths remain for compatibility. The example shell script targets ImageNet and is not a validated bone-age reproduction command.

### Permissions and remaining reproduction limits

Download from original providers and follow their terms. RSNA permits academic/educational and other non-commercial use subject to its [terms and attribution requirements](https://www.rsna.org/artificial-intelligence/ai-image-challenge/rsna-pediatric-bone-age-challenge-2017); cite Halabi et al., *The RSNA Pediatric Bone Age Machine Learning Challenge*, DOI: 10.1148/radiol.2018180736. For RHPE, review the terms linked on BAAR and cite Escobar et al., *Hand Pose Estimation for Pediatric Bone Age Assessment*, MICCAI 2019. DHA access and redistribution conditions must be confirmed with its provider. This repository's code license does not grant rights to third-party datasets.

No annotation files are redistributed by this update. Exact historical splits, DHA preprocessing/label aggregation, the original sex encoding, and test-label provenance remain unconfirmed. Syntax and synthetic crop checks do not establish full model or published-result reproducibility; real-data verification requires authorized data and the appropriate runtime dependencies.
