# Data Directory

Place your dataset here or point the config to an external location.

Expected structure for the default template:

```text
data/
  candidate/images/
  labeled/images/
  unlabeled/images/
  val/images/
  annotations/
    candidate_train.json
    coverage_head_labeled.json
    labeled_train.json
    unlabeled_train.json
    val.json
    test.json
```

Annotation files should use COCO detection format.

`candidate_train.json` is the image list used by cold-start selection.
`coverage_head_labeled.json` is only needed when running the cluster-tail
selector in a live annotation workflow.
