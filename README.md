# revgrad-learningemotion
Just a school final project on domain adaptation *(FER2013 to Student learning emotion).


## Dataset

### FER2013
We roughly convert the basic emotion labels of the dataset into learning emotion labels, then put whole images in `./fer2013` like:
```
.
├── test
│   ├── 0_frustrated
│   ├── 1_confuse
│   ├── 2_boring
│   ├── 3_happy
│   ├── 4_flow
│   └── 5_surprise
└── train
    ├── 0_frustrated
    ├── 1_confuse
    ├── 2_boring
    ├── 3_happy
    ├── 4_flow
    └── 5_surprise
```
* Write an `annotation.csv` file in `./annotations/` for our custom dataset class.

### Student Emotion

The dataset we collected ourselves in this experiment.