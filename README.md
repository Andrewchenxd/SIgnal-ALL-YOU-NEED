

# SIG-ALL-YOU-NEED
信号领域论文复现（不定期更新）


**RML数据集生成**

数据集生成代码参考：
```
@ARTICLE{10070586,
  author={Sathyanarayanan, Venkatesh and Gerstoft, Peter and Gamal, Aly El},
  journal={IEEE Transactions on Wireless Communications}, 
  title={RML22: Realistic Dataset Generation for Wireless Modulation Classification}, 
  year={2023},
  volume={22},
  number={11},
  pages={7663-7675},
  keywords={Modulation;Wireless communication;Mathematical models;Atmospheric modeling;Computational modeling;Benchmark testing;Ad hoc networks;Deep learning;modulation classification;benchmark dataset;GNU radio;spectrum sensing},
  doi={10.1109/TWC.2023.3254490}}
```
##### **Linux环境下安装依赖库的命令** :
```
1. conda create --name gnuradio
2. conda activate gnuradio
3. conda install -c conda-forge gnuradio=3.8.3
4. conda install -c conda-forge scipy
5. conda install -c conda-forge matplotlib
6. git clone https://github.com/myersw12/gr-mapper.git
7. cd gr-mapper && mkdir build && cd build
8. chmod -R 777 ../../
9. conda install -c conda-forge gnuradio-build-deps
10. conda activate $CONDA_DEFAULT_ENV
11. conda install -c conda-forge cppunit
12. cmake -G Ninja -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DLIB_SUFFIX="" ..
13. cmake --build .
14. cmake --build . --target install

```
Podcast.wav下载链接[link](https://drive.google.com/drive/folders/1dEv6gPwPahUfFFRYYxvp3i5M34D7KI9J?usp=sharing)；

下载并放到"generate_data/Podcast.wav"目录下；

RML22数据集与RML16数据集在代码生成上的主要不同之处：
```
samples_per_symbol=2 #RML22
samples_per_symbol=8 #RML16
```


### **Citation**
If you find this repository useful in your work, please consider citing the following paper:
```bash
@article{feng2024learning,
  title={Learning temporal--spectral feature fusion representation for radio signal classification},
  author={Feng, Zhixi and Chen, Shuai and Ma, Yue and Gao, Yachen and Yang, Shuyuan},
  journal={IEEE Transactions on Industrial Informatics},
  year={2024},
  publisher={IEEE}
}
```

```bash
@article{chen2024generative,
  title={A Generative Self-supervised Framework for Cognitive Radio Leveraging Time-Frequency Features and Attention-based Fusion},
  author={Chen, Shuai and Feng, Zhixi and Yang, Shuyuan and Ma, Yue and Liu, Jun and Qi, Zhuoyue},
  journal={IEEE Transactions on Wireless Communications},
  year={2024},
  publisher={IEEE}
}
```

```bash
@article{chen2025radiollm,
  title={RadioLLM: Introducing Large Language Model into Cognitive Radio via Hybrid Prompt and Token Reprogrammings},
  author={Chen, Shuai and Zu, Yong and Feng, Zhixi and Yang, Shuyuan and Li, Mengchang and Ma, Yue and Liu, Jun and Pan, Qiukai and Zhang, Xinlei and Sun, Changjun},
  journal={arXiv preprint arXiv:2501.17888},
  year={2025}
}
```



