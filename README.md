# RoadGarbage-KV260

基于 KV260 与 YOLOv5s 的道路垃圾识别及激光指向系统。

## 主要功能
- 垃圾检测与分类
- 双目目标匹配与稳定确认
- 实际坐标转换
- 激光振镜指向
- 网页交互与结果显示

RoadGarbage-KV260/
├── README.md                         项目介绍、运行方法、硬件来源
│
├── hardware/                         硬件模块源码
│   ├── dac_1.0/                       DAC IP 文件夹
│
├── rubbish_laser/                    板端运行文件
│   ├── run_rubbish_demo_ourdata.py
│   ├── run_rubbish_demo_ui.html
│   ├── general.py
│   ├── calibration_manifest_ourdata.json
│   ├── garbage_yolov5_ourdata.xmodel
│   ├── dpu.bit
│   ├── dpu.hwh
│   └── dpu.xclbin
│
└── docs/                             项目展示材料
    └── road_litter_poster_key_tech.pptx