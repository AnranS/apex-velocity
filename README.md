# 极速之巅 · APEX VELOCITY

一款运行在浏览器里的 3D 赛车游戏（H5）。Three.js + WebGL2 渲染，自研车辆物理与 AI，程序化音频。

## 运行

```bash
npm install
npm run dev        # 开发模式：http://localhost:5199/
npm run build      # 生成静态站点到 dist/
npm run preview    # 本地预览 dist/：http://localhost:4173/
```

`dist/` 是纯静态文件，可部署到任意静态服务器或 CDN（需通过 http/https 访问，不能直接双击 index.html）。

## 操作

键位与漂移逻辑参照 QQ飞车端游的默认设置。

| 动作 | 键盘 | 手柄 |
|---|---|---|
| 加速 / 刹车·倒车 / 转向 | `↑` `↓` `←` `→`（也可用 `W` `S` `A` `D`） | RT / LT / 左摇杆 |
| 漂移 | `Shift` + 方向键 | A |
| 使用氮气 | `Ctrl`（Mac 可用 `空格`） | X / RB |
| 复位 | `R` | Back |
| 切换视角 / 回头看 / 暂停 | `C` / `B` / `Esc` | Y / LB / Start |

- **漂移**：按住方向键再按 `Shift` 起漂，按住 `Shift` 角度越甩越大；松开后车仍在滑行，反打方向"拉车头"让车身回正，漂移才结束。
- **小喷 / 双喷**：车身回正的瞬间松开再按一下 `↑` 出小喷；小喷后立刻再点一下漂移并补 `↑` 就是双喷。
- **氮气**：漂移集气，集满得到一瓶（最多两瓶），按 `Ctrl` 喷射；喷射中接小喷可叠加提速。
- **超级起步 / 落地喷**：绿灯亮起瞬间点一下 `Shift`；腾空落地瞬间按 `↑`。

触屏设备会自动显示屏幕按键（自动油门，小喷自动触发）。macOS 上 `Ctrl` + 方向键是系统快捷键（调度中心 / 切换桌面），建议用 `空格` 放氮气，或改用 `WASD` 驾驶。

## 网址参数

`?race=1` 跳过菜单直接开赛，可叠加 `w=sunset|day|storm|night`、`car=aurion|valken|zephyr`、`laps`、`ai`、`diff=0|1|2`、`q=low|medium|high|ultra`、`bot=1`（AI 代驾）、`hud=0`（隐藏界面）。

## 目录

- `src/sim/` 赛道几何、车辆动力学、AI、比赛规则（不依赖 DOM，可在 Node 里跑：`npm run simtest`）
- `src/render/` 渲染引擎、天空与天气、赛道世界、场景、车辆外观、特效
- `src/audio/` 程序化音频（引擎合成、音效、音乐）
- `src/ui/` 菜单与 HUD
- `tools/` 资产生成脚本（Blender / Python）与自动化测试脚本
- `public/assets/` 模型、贴图、天空

## 素材来源

车辆、赛道设施、植被模型为本项目用脚本生成。以下素材来自 [Poly Haven](https://polyhaven.com)（CC0）：
天空 HDRI（kloppenheim_06_puresky、kloofendal_48d_partly_cloudy_puresky、overcast_soil_puresky、qwantani_moonrise_puresky），
地表贴图（asphalt_track、leafy_grass、rock_face_03、gravel_floor_02、forest_ground_04、coast_sand_01、concrete_floor_02），
以及环境资产包中的树皮贴图与岩石扫描（详见 `public/assets/env/manifest.json` 的 `credits`）。
