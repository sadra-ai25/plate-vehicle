LABEL author and manager = "Ali Hadipour <a.hadipour64@gmail.com>"
LABEL colleague = "Ali Tavakolpour <ali.tavakolpoorsaleh@gmail.com>"
Company name = "Sadraafzar <https://sadraafzar.com/>"

This version is final version for Foolad Factory production.
    version number: v1.0

The accuracy of datasets:
{
    Train dataset: 99.87
    Val dataset: 88.3
}

The release time of application is: 2025.01.23 / 1403.11.04

The weights path is: plate-server/app/weights
    best.pt: yolov5 method - release: 2025.01.15 / 1403.10.26
    best.chpt: crnn method - release: 2025.01.23 / 1403.10.26

The dataset path is: ~/.../datasets/plate 

├── 0_produced
│   ├── detection
│   ├── orig-plate
│   └── recognition
├── 1_foolad
│   ├── detection
│   ├── orig-plate
│   └── recognition
├── 2_hooshmand
│   ├── detection
│   ├── orig-plate
│   └── recognition
├── 3_malek
│   ├── detection
│   ├── orig-plate
│   └── recognition
└── all_images
    ├── detection
    ├── orig-plate
    └── recognition
datasets included 0_produced and 1_foolad
