|                   mode                 |n_gpu|  btz |    gpu    |    mem  |
|                   ----                 |---- | ---- |   ----    |   ----  |
|                  torch                 |  1  |  16  |   11.6g   |   1.7g  |
|                ds+stage1               |  1  | 1*16 |   10.1g   |   1.9g  |
|                ds+stage2               |  1  | 1*16 |   12.8g   |   1.9g  |
|                ds+stage3               |  1  | 1*16 |   14.3g   |   2.0g  |
|              ds+fp16+stage1            |  1  | 1*16 |    6.4g   |   2.0g  |
|              ds+fp16+stage2            |  1  | 1*16 |    7.7g   |   1.9g  |
|              ds+fp16+stage3            |  1  | 1*16 |    7.2g   |   2.0g  |
|        ds+fp16+stage1+offload_opt      |  1  | 1*16 |    5.5g   |   4.8g  |
|        ds+fp16+stage2+offload_opt      |  1  | 1*16 |    7.3g   |   4.8g  |
|        ds+fp16+stage3+offload_opt      |  1  | 1*16 |    ———    |    ———  |
| ds+fp16+stage3+offload_opt+offload_para|  1  | 1*16 |    6.4g   |   5.3g  |
|                ds+stage1               |  2  | 2*8  | 5.9g+5.1g |2.4g+2.0g|
|                ds+stage2               |  2  | 2*8  | 10.2g+9.5g|2.4g+2.0g|
|                ds+stage3               |  2  | 2*8  | 9.4g+8.6g |2.4g+2.0g|
|              ds+fp16+stage1            |  2  | 2*8  | 4.5g+3.9g |2.4g+2.0g|
|              ds+fp16+stage2            |  2  | 2*8  | 6.6g+5.9g |2.4g+2.0g|
|              ds+fp16+stage3            |  2  | 2*8  | 5.9g+5.1g |2.4g+2.0g|
|        ds+fp16+stage1+offload_opt      |  2  | 2*8  | 5.1g+4.2g |4.5g+4.1g|
|        ds+fp16+stage2+offload_opt      |  2  | 2*8  | 6.0g+5.2g |4.5g+4.1g|
|        ds+fp16+stage3+offload_opt      |  2  | 2*8  |    ———    |   ———   |
| ds+fp16+stage3+offload_opt+offload_para|  2  | 2*8  | 5.7g+4.5g |4.7g+4.2g|
