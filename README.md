# 苗圃遮阳台传感器缺测补记

河桥苗圃有 12 名育苗与移栽工作人员，使用两种开孔率的遮阳网台架。轮换探头时，温度、湿度或光照读数可能漏录。种植员在排移栽班前需要知道哪些读数有参考值，哪些仍需回台补测。

## 启动与登录

```sh
cp .env.example .env
docker compose build backend
docker compose up -d --no-build --wait
curl -fsS http://127.0.0.1:19439/api/health
curl -fsS http://127.0.0.1:19439/api/spec
```

Python 3.11 FastAPI 处理请求，MySQL 8.0 保存批次、运行、签核与审计，Redis 唤醒后台 worker。数据库只绑定本机 `17939`。启动时默认加载合成数据，可通过 `SEED_DEMO=false` 停用，也可手动运行 `docker compose exec -T backend python -m scripts.seed_demo`。

| 演示账号 | 密码 | 操作 |
|---|---|---|
| `analyst@demo.local` | `Demo-2611!` | 导入读数、试算参数、检查插补或拒补原因 |
| `admin@demo.local` | `Demo-2611!` | 查阅审计记录、签核交接单 |
| `viewer@demo.local` | `Demo-2611!` | 查看台账与获批结果、下载 CSV |

使用 `POST /api/auth/login` 获取令牌，随后发送 `Authorization: Bearer <token>`。CSV 完整列名见 `GET /api/spec`，OpenAPI 见 `/api/openapi.json`。CSV 简例仅说明格式，正式运行还需要足够的训练和留出记录。

```csv
reading_id,batch_id,partition,shade_class,temp_c,rh_pct,par
TR-01-01,TR-01,train,mesh30,19.4,78.3,128.2
CA-01-01,CA-01,calib,mesh30,19.5,78.0,126.7
HO-01-01,HO-01,holdout,mesh30,19.7,77.9,133.1
PE-01-01,PE-01,pending,mesh30,19.8,78.1,
```

1. `GET /api/datasets` 获取演示台账，或 `POST /api/datasets` 提交 `{"name":"移栽测次","csv":"…"}`。
2. `POST /api/runs` 提交 `{"label":"二号遮阳台上午补测","dataset_id":1,"settings":{}}`，记录返回的运行 ID。
3. `GET /api/runs/{id}` 查看覆盖率、每个字段的留出 MAE、缺测组合的九成联合区间及建议值。
4. 复核员通过 `POST /api/runs/{id}/review` 提交 `{"decision":"approved","note":"已核对补测清单"}`。
5. 从 `GET /api/runs/{id}/export/missing_cells` 下载交班表。字段误差、遮挡明细、组合区间、校准误差、整批留出覆盖和冻结尺度各有独立导出表。

## 计算与数据范围

同档训练读数的标准差用于缩放目标字段之外的有效已测列。算法取标准化均方根距离最小的 `neighbors` 条记录，按距离倒数加权。如果有零距离记录，就取所有零距离供体的均值。可用字段数低于 `min_anchors` 或同档记录不足时，导出人工补测原因。`mask_seed` 决定留出批遮挡顺序。每行隐藏一个字段，KNN 与同档训练中位数在相同可估单元上按字段对比 MAE。单列随机遮挡的覆盖率只适用于此回测设定。

### 缺测组合的九成联合区间

待补单元按遮阳网档位与缺测字段组合（如 `temp_c+par`）分组，每组只使用 `calib` 批校准、`holdout` 批验证，校准和留出读数都不作为近邻供体。

1. 对每个校准批，遮挡组内的缺测字段，取该批所有单元标准化误差（误差/同档训练尺度）的最大值作为该批得分。
2. 九成联合区间半径取这些批次得分的 conformal 分位数 `ceil((n+1)*0.9)`，需至少 9 个可用校准批；不足时该组标记 `校准批不足`，导出应补校准批数。
3. 在独立留出批遮掉相同组合，整批所有单元都落在区间内才算覆盖；留出批出现无法给出建议的单元也计为未覆盖。
4. 交班表的建议值随附 `joint_low`、`joint_high`、`coverage_status` 与 `holdout_coverage`。

晚到的校准数据只参与新一轮试算。已签核的交班表继续从原运行结果下载当时的区间和依据；要采用新校准数据，需重新提交运行并重新签核。

数据要按 `train`、`calib`、`holdout`、`pending` 整批分区，定义在 `backend/src/domain/settings.py`，并由 `records.py` 校验。`calib` 与 `holdout` 是彼此独立的完整读数，`pending` 才允许缺测。`Role`、`RunStatus` 和 `Decision` 定义于 `backend/src/constants/workflow.py`，用于鉴权、worker 状态与签核接口。业务代码位于 `backend/src/domain`，应用层位于 `backend/src/{routes,services,models,schemas}`，数据库初始化与 seed 位于 `backend/migrations`、`backend/scripts`。`.env.example` 包含端口、签名密钥及演示数据开关。

第三方依赖采用各自的许可证。
