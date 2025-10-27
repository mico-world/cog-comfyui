# Your repo
## 拉取代码
连同子仓库一起拉取: `git clone --recurse-submodules` 

## 安装依赖
pip install -r requirements.txt

## 添加三方节点
python scripts/add_custom_node.py 

将控制台新增的依赖添加到requirement.txt去除重复依赖

## 下载三方节点
将三方节点clone到ComfyUI文件夹
python scripts/install_custom_nodes.py 

## 启动comfyui
sh scripts/start.sh     