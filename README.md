# Speech-Understanding-Minor-Project

## Description
This project is build intended to submit as Minor Project for Speech Understanding class at IIT, Jodhpur. 

## Project Setup and Uses Steps
### Installation
To install and run this project, follow these steps:

1. Clone the repository: `git clone https://github.com/parasharharsh16/Speech-Understanding-Minor-1.git`
2. Navigate into the project directory: `cd Speech-Understanding-Minor-1`
3. Install the required dependencies: `pip install -r requirements.txt`

### Prepare Dataset
To download the "MUSAN" dataset, please follow the below link:
`https://www.openslr.org/resources/17/musan.tar.gz`

- Unzip the downloaded TAR file and paste the musan dataset to "data" folder.
```
Speech-Understanding-Minor/
│
├── code/
│   ├── main.py
│   ├── dataloader.py
│   ├── utils.py
│   ├── param.py
│   └── modelarchitecture.py
│
├── datafolder/
│   └── Musan/
│       ├── music/
│       ├── speech/
│       └── noise/
│
├── model/
│   └── trainedmodel.pth
│
└── README.md
```

### Run the code
4. Go to code/param.py and change `train_model` to  `False` to run in evaluation mode, if intending to train then you may keep it to `True`
5. Run the main script: `python main.py`

P.S. Do not move to code folder to run the project


## Team Members

- Prateek Singhal (m22aie215@iitj.ac.in)
- Prabha Sharma (m22aie224@iitj.ac.in)
- Harsh Parashar (m22aie210@iitj.ac.in)
