# Data Directory

Place your dataset here or point the config to an external location.

Expected structure for the default template:

```text
data/
  labeled/images/
  unlabeled/images/
  val/images/
  annotations/
    labeled_train.json
    unlabeled_train.json
    val.json
```

Annotation files should use COCO detection format.
