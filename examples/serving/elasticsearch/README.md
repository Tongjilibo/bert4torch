## 使用说明
### 1. 启动elastic search
- elastic search使用是7.3.2版本, 先启动elastic search
- 我使用的window版本，[下载链接](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.3.2-windows-x86_64.zip)

### 2. 插入index + doc转emb + 插入数据 + 开启服务
```bash
sh start.sh
```

### 3. Open browser
- Go to <http://127.0.0.1:5000>.

## 参考资料
- [bertsearch](https://github.com/Hironsan/bertsearch), 拉下来不能完全跑通，修改了文件权限和Flask版本后才跑通