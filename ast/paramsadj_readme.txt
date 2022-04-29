这个模型需要在Linux系统下跑 需要gpu 可以用great lakes或者其他云计算平台

解压ast.zip，前往egs/audioset

第一次运行run.sh可能会有权限问题，先运行chmod -R 777 run.sh

运行./run.sh可以训练模型并且在evaluation set上validate，结果看输出就行，主要看average precision


调参：（更改run.sh中的变量值）

默认运行25个epoch，batch size 12（不建议更改），跑完一个epoch时间应该小于300秒

可以更改lr, mixup, fstride（默认10，表示频率方向上的步长）, tstride（默认10，表示时间方向上的步长）

默认patch size 16*16（stride = 10说明相邻patch有重合）


