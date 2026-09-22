# 寒武纪 MLU · 镜像获取渠道调研（第 3 家接入前置）

> 日期：2026-09-22 ｜ 负责人：Kistich（hliu553）｜ 方向：设备抽象与执行上下文
> 目的：回答"寒武纪这一支的镜像从哪来、怎么选"，供**组内基座确定**与**向寒武纪申请资源**使用
> 方式：官方文档核查（FlagTree wiki 全页 / 寒武纪开源仓库）+ 两台测试机上的**只读网络探测**

---

## 1. 结论速览

| 问题 | 结论 |
|---|---|
| FlagTree 有寒武纪的官方 User Manual / 推荐镜像吗？ | ❌ **没有**（wiki 全部 26 页里无 cambricon/mlu 条目；其他 17 个后端都有） |
| 寒武纪有官方镜像吗？ | ✅ **有**，但**必须走官方渠道申请**（社区账号 / 镜像 tarball / 私仓凭据） |
| 官方镜像仓可达吗？ | ✅ 三处**均可从测试机访问**，但**都返回 401 需鉴权**，且都已列入本机 `daemon.json` 的 `insecure-registries` |
| 推理形态（与昇腾/昆仑芯哪个同类）？ | **与昇腾同类** —— 寒武纪有**厂商移植版 vLLM**（官方开源 `Cambricon/vllm-mlu`），不是"社区 vLLM + 第三方平台插件" |
| 我们现在能自己拉吗？ | ❌ **不能** —— 无仓凭据、且不在 `docker` 组（两道都需开通） |

**⇒ 与前两家最大的不同：**910C 与 P800 的镜像都能从**公开上游**拿到（华为 quay / BAAI Harbor），
**寒武纪必须走官方渠道**。这是第 3 家接入的**头号环境风险**，应在申请机器时一并提出。

---

## 2. FlagTree 侧：确认「没有」寒武纪手册（附证据）

FlagTree wiki 的 **Pages 索引共 26 页**，逐页列出如下（`https://github.com/flagos-ai/FlagTree/wiki/_pages`）：

```text
Home · Architecture · FlagTree Backend Specialization · Hints · Project Info · TLE · TLE Raw ·
Triton Version Upgrade Report · User Manual ·
User manual for: aipu · amd · ascend · cpu · enflame · hcu · iluvatar · metax · mthreads ·
                 nvidia · ppu · rpu · spacemit · sunrise · tileir · tsingmicro · xpu
```

- **有** `User-manual-for-xpu`（昆仑芯）、`User-manual-for-ascend`（昇腾）、`User-manual-for-ppu`（平头哥）等；
- **没有** `User-manual-for-cambricon`（直接访问该路径会重定向回 wiki 首页 ⇒ 页面不存在）。

而 **User Manual 索引页正文**写着：*"The best practice to avoid environment compatibility issues is to
use the image mentioned in the backend documents above"* —— 索引页的安装表覆盖
nvidia / tileir / amd / aipu / ascend / enflame / hcu / iluvatar / metax / mthreads / ppu / sunrise /
tsingmicro / xpu，**同样没有 cambricon 行**。

**寒武纪在 FlagTree 里确实存在**，但只在编译器侧：`Project-Info` 的分支表写明
**`triton_v3.2.x` 分支覆盖 NVIDIA / AMD / Huawei Ascend / **Cambricon****（Triton 3.2）；
即**有 triton 后端、无镜像与使用手册**。

> ⇒ 结论：**寒武纪这一支没有 FlagTree 官方推荐镜像**。这一点要在向总组提交材料时讲清楚，
> 否则会被问"为什么不用官方推荐镜像"（前两家我们都被问过同类问题）。

---

## 3. 寒武纪官方侧：渠道与镜像仓（含实测）

### 3.1 官方渠道

| 渠道 | 说明 |
|---|---|
| **寒武纪开发者社区** `developer.cambricon.com` | 权威入口：官方开源仓库 `Cambricon/torch_mlu` 的 README 在"**版本配套关系**"表里，把**镜像**一列直接指向该社区 |
| **官方镜像仓**（3 个，见 §3.2） | 需凭据；已在本机 `daemon.json` 的 `insecure-registries` 中预置 |
| **镜像 tarball** | 官方文档示例以 `docker load -i <名>.tar.gz` 形式分发（如 `cambricon_vllm_container.tar.gz`） |

### 3.2 三个 registry 的实测（在两台测试机上探测，只读）

| registry | DNS | 端口 | `/v2/` 响应 | 鉴权类型 |
|---|---|---|---|---|
| `docker.cambricon.com` | `10.1.3.46` | **80 开放**（443 关闭） | **401** | `Bearer realm="https://docker.cambricon.com:5001/auth", service="Docker registry"` ⇒ **自建 Docker Distribution** |
| `docker-user.cambricon.com:30080` | `10.1.2.14` | **30080 开放** | **401** | `Bearer realm="http://.../service/token", service="harbor-registry"` ⇒ **Harbor** |
| `docker-user.extrotec.com:30080` | `222.190.151.165`（公网） | **30080 开放** | **401** | 同上（Harbor） |

- 三处 `_catalog` 同样 **401** ⇒ **匿名不可用**，必须有账号；
- 两台测试机的 `/etc/docker/daemon.json` **已把这三处列入 `insecure-registries`**
  （即走 HTTP，故 https 端口关闭属正常）。

### 3.3 官方镜像命名规律（公开实证样本）

从寒武纪官方文档/公开材料取得的实际镜像名（**用于识别与申请时对齐口径**，非我们的目标版本）：

```text
cambricon/pytorch:v24.12-torch2.5.0-torchmlu1.24.0-ubuntu22.04-py310
cambricon-base/pytorch:v25.01-torch2.5.0-torchmlu1.24.1-ubuntu22.04-py310
cambricon_pytorch_container-torch2.7.1-torchmlu1.28.0-ubuntu22.04-py310.tar.gz
cambricon_vllm_container.tar.gz                      ← 推理用（对应 Cambricon/vllm-mlu）
```

规律：`<厂商>/pytorch:<SDK版本>-torch<X>-torchmlu<Y>-ubuntu<Z>-py<P>`，
其中 `-base` 前缀变体（`cambricon-base/pytorch:...`）对应基础镜像。

### 3.4 我们需要的目标版本档

**依据**：测试机 `/srv/data/build_base_venv.sh`（他人（`x-benchmark` 负载）的资产，只读可读）标题为
"寒武纪 MLU 基础 venv 重建（torch-mlu + triton-mlu + flag_gems）"，其中给出的组合为：

```text
python           3.12
torch            2.11.0+cpu
torch-mlu        1.33.1+torch2.11.0
torch-mlu-ops    1.12.1+torch2.11.0
triton           3.4.0+mlu2.1.1
NEUWARE_HOME     /usr/local/neuware
私有 wheel 源    https://resource.flagos.net/repository/flagos-pypi-cambricon/simple
```

⇒ 对应的官方镜像档位应为 **`torch2.11.0` + `torchmlu1.33.1` + `ubuntu22.04` + `py312`**
（⚠️ **确切 tag 待寒武纪方确认** —— 我们无法从公网列出该私仓的 tag 清单，**不臆造 tag**）。

---

## 4. 需要向寒武纪 / 管理员申请的事项（与机器开通一并提出）

| # | 事项 | 用途 | 备注 |
|---|---|---|---|
| 1 | **镜像仓凭据**（`docker.cambricon.com` 或 `docker-user.cambricon.com:30080` 的账号）**或** 直接提供镜像 tarball | 拿到含 `torch_mlu` + NeuWare 的容器 | 两选一即可；tarball 更省事（无需凭据） |
| 2 | **确切镜像 tag**（对齐 `torch2.11.0` / `torchmlu1.33.1` / `py312` 档） | 归档与入锁材料要写 tag+digest | 我们列不出私仓 tag，需厂商给 |
| 3 | **容器内是否含 vLLM 移植版**（`Cambricon vLLM Container`） | 推理腿形态（对齐昇腾 `vllm-ascend` 的做法） | 若没有，则推理腿退化为"社区 vLLM + 手工装 `vllm-mlu`" |
| 4 | 与已登记的 root 权限需求配套：`/srv/hliu553` + `docker` 组 | 落数据、起带卡容器 | 见 `../README.md` §4 |

---

## 5. 三家实例的镜像获取路径对照（供基座确定口径）

| | 第 1 家 昇腾 910C | 第 2 家 昆仑芯 P800 | **第 3 家 寒武纪 MLU590** |
|---|---|---|---|
| 公开上游推荐镜像 | ✅ **有**：FlagTree `User-manual-for-ascend` | ✅ **有**：FlagTree `User-manual-for-xpu` | ❌ **无**（FlagTree 无寒武纪手册） |
| 推荐镜像所在仓 | BAAI Harbor（`harbor.baai.ac.cn/flagtree/...`）；推理另有华为 `quay.io/ascend/vllm-ascend` | BAAI Harbor（`harbor.baai.ac.cn/flagtree/...:202608-base`） | 寒武纪私仓（`docker.cambricon.com` / `docker-user.cambricon.com:30080`）+ 开发者社区 |
| 获取难度 | 低（公开可拉） | 低（公开可拉，本机已有） | **高（需申请凭据或 tarball）** |
| 推理形态 | 厂商移植版 vLLM（`vllm-ascend`） | 社区 vLLM + 第三方平台插件（`vllm-plugin-FL`） | **厂商移植版 vLLM**（`Cambricon/vllm-mlu`）—— 与昇腾同类 |
| 本方向现状 | 已入锁（训练腿 + 推理腿各一） | 在位、有 digest，**建议以官方 `-base` 入锁** | **待申请** |

---
 
## 6. 待验证 / 未取得（如实标注）

- ⚠️ **确切镜像 tag 与 digest 未取得** —— 私仓需鉴权，无法列 tag；**不猜测、不补零**；
- ⚠️ 三个私仓**是否都提供同一套镜像**未验证（可能是"官方仓 + 渠道仓"分层，权限不同）；
- ⚠️ `docker-user.extrotec.com` 解析到公网 IP，**从本机可达性已实测为开放**，但其与寒武纪官方的授权关系未确认；
- ⚠️ 官方镜像内**是否已含 vLLM 移植版**未验证（§4 第 3 项）；
- ⚠️ FlagTree `triton_v3.2.x` 分支的 cambricon 后端**与本方向职责无关**（那是编译器/算子侧），
  但若后续算子方向要在寒武纪上跑 triton，这条线需要另议 —— 本报告不涉及。

## 7. 附：一键复核命令（后续任何人可复跑）

```bash
# 三个 registry 的可达性与鉴权要求（在任何能访问该网段的机器上执行）
for R in docker.cambricon.com docker-user.cambricon.com:30080 docker-user.extrotec.com:30080; do
  echo "---- $R"
  curl -sI -m 12 "http://$R/v2/" | grep -iE '^HTTP|www-authenticate'
done

# FlagTree wiki 页面清单（确认有无 cambricon 手册）
curl -sL https://github.com/flagos-ai/FlagTree/wiki/_pages | grep -oiE 'User-manual-for-[a-z]+' | sort -u
```
