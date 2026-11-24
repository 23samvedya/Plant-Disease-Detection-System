# Example images

Sample apple photos for trying the web app (they were originally saved by the app into `uploads/`).

Most of them are copies of dataset images: [`index.csv`](index.csv) lists, for each file, whether it is
in the `train`, `val` or `test` split, or `not_in_dataset` (out of 63 files: 39 train, 4 val, 11 test,
9 not in the dataset). A correct prediction on a `train` image is **not** evidence that the model
generalises; use the `test` or `not_in_dataset` images for that. Some images come from web image
searches, so check their licences before reusing them elsewhere.
