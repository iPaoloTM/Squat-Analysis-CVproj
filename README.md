# A Case Study for the Automatic Supervision of Body-Weight Exercises: The Squat

Official code for the iSTAR 2023 paper "A Case Study for the Automatic Supervision of Body-Weight Exercises: The Squat". 
This repository contains the official python implementation of the experiments described in the paper.

![zed_t](https://github.com/iPaoloTM/Squat-Analysis-CVproj/assets/43711362/11ee522f-e720-4be2-9e0d-817997777ad8)

 3D body34 model in T-pose (captured with ZED2)
 <hr>
 
![Figure_1](https://github.com/iPaoloTM/Squat-Analysis-CVproj/assets/43711362/5032f890-d2dd-4d7a-a26d-6129acfb8ed0)

MOCAP Skeletons in deep squat positions aligned with Procrustes transformation: correct squat (blue) and bad squat (red)

## Usage

The code is originally developed in Python 3.7.

````
pip install <libraries>

python ZED/ZED_3Dplot.py referenceZED 340

python MOCAP/MOCAP_3Dplot.py referenceMOCAP 3400

````

## Citation

Please cite the following paper if you use this code directly or indirectly in your research/projects:

```
@inproceedings{SquatAnalysis:iSTAR:2023,
  title = {A Case Study for the Automatic Supervision of Body-Weight Exercises: The Squat},
  author = {Aliprandi, Girardi, Martinelli, De Natale, Bisagno and Conci},
  booktitle = {International Workshop on Sport Technology and Research (IEEE-STAR)},
  month = jun,
  year = {2023},
  doi = {},
  month_numeric = {06}}
```

## License

Software Copyright License for **non-commercial scientific research purposes**. Please read carefully
the [terms and conditions](./LICENSE) and any accompanying documentation before you download and/or
use the SOMA data and software, (the "Data & Software"), software, scripts, and animations. 
By downloading and/or using the Data & Software (including downloading, cloning, installing, and any other use of this repository), 
you acknowledge that you have read these terms
and conditions, understand them, and agree to be bound by them. If you do not agree with these terms and conditions, you
must not download and/or use the Data & Software. 
Any infringement of the terms of this agreement will automatically terminate
your rights under this [License](./LICENSE).

## Contact

The code in this repository is developed by [Paolo Aliprandi](https://github.com/iPaoloTM) and [Letizia Girardi](https://github.com/letiziagirardi)
while at [University of Trento](https://www.unitn.it).

If you have any questions you can contact us at [mmlab-disi@unitn.it](mailto:mmlab-disi@unitn.it).

For commercial licensing, contact [mmlab-disi@unitn.it](mailto:mmlab-disi@unitn.it)
