import torch
from dpt.models_with_adabins import DPTDepthModelWithAdaBins

# 创建模型
model_path = None
model = DPTDepthModelWithAdaBins(
    path=model_path,
    backbone="vitb_rn50_384",
    non_negative=True,
    enable_attention_hooks=False,
)

# 定义新模型的权重字典
model_dict = model.state_dict()
# 打印新模型的权重字典
for key, value in model_dict.items():
    print(key)
print('---------------------')

# 定义AdaBins预训练模型权重字典
pretrained_dict_adabins = torch.load(r"E:\MasterProject\AdaBins-main\pretrained\AdaBins_nyu.pt")
# 打印AdaBins预训练模型权重字典
# for key, value in pretrained_dict_adabins.items():
#     print(key)
# 获取AdaBins预训练模型权重字典（包含了model，optimizer，和epoch）中"model"的权重字典
model_weights_adabins = pretrained_dict_adabins.get('model', {})

'''筛选adabins预训练权重模型'''
# 开始筛选AdaBins预训练权重模型使得能与新模型权重字典相匹配
model_weights_adabins_filtered = {}
# 仅保存key中带有module.adaptive_bins_layer和module.conv_out的键值对，并且删除module.的前缀
for key, value in model_weights_adabins.items():
    if "module.adaptive_bins_layer" in key or "module.conv_out" in key:
        new_key = key[len("module."):]
        model_weights_adabins_filtered[new_key] = value
# 打印检查筛选后的预训练权重模型字典
for key, value in model_weights_adabins_filtered.items():
    print(key)
print('----------------')

# 此处检查发现AdaBins预训练权重字典和新模型权重字典有几个keys名字不匹配，此处为了完全匹配，选择更改预训练权重字典如下keys的名字
key_mapping = {
    "adaptive_bins_layer.patch_transformer.embedding_encoder.weight": "adaptive_bins_layer.patch_transformer.embedding_convPxP.weight",
    "adaptive_bins_layer.patch_transformer.embedding_encoder.bias": "adaptive_bins_layer.patch_transformer.embedding_convPxP.bias",
    "adaptive_bins_layer.embedding_conv.weight": "adaptive_bins_layer.conv3x3.weight",
    "adaptive_bins_layer.embedding_conv.bias": "adaptive_bins_layer.conv3x3.bias",
}
new_model_weights_adabins_filtered = {}
for key, value in model_weights_adabins_filtered.items():
    if key in key_mapping:
        new_key = key_mapping[key]
        new_model_weights_adabins_filtered[new_key] = value
    else:
        new_model_weights_adabins_filtered[key] = value

test = {}
for key, value in model_dict.items():
    if "adaptive_bins_layer" in key or "conv_out" in key:
        test[key] = value
        if key not in new_model_weights_adabins_filtered.keys():
            print(f"{key} not in new_model_weights_adabins_filtered.keys()")

# 将两个权重字典中的所有权重都移至 cuda:0
new_model_weights_adabins_filtered_cuda = {k: v.to('cuda:0') for k, v in
                                             new_model_weights_adabins_filtered.items()}
test_cuda = {k: v.to('cuda:0') for k, v in test.items()}

def compare_dicts(dict1, dict2):
    # 比较两个字典的键
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    return True

# AdaBins预训练权重字典和新模型权重字典完全匹配
print(f"Are all key-value pairs match? {compare_dicts(test_cuda, new_model_weights_adabins_filtered_cuda)}")
print(len(test_cuda) == len(new_model_weights_adabins_filtered_cuda))

'''开始筛选dpt模型'''
# 定义DPT预训练模型权重字典
pretrained_dict_dpt = torch.load('E:\MasterProject\DPT-main\weights\dpt_hybrid-midas-501f0c75.pt')
# 打印DPT预训练模型权重字典
for key, value in pretrained_dict_dpt.items():
    print(key)
print('---------------------')

# 开始筛选dpt预训练权重文件
pretrained_dict_dpt_filtered = {}
for key, value in pretrained_dict_dpt.items():
    if "scratch.output_conv" in key:
        continue
    pretrained_dict_dpt_filtered[key] = value

# 检查长度
print(len(pretrained_dict_dpt), len(pretrained_dict_dpt_filtered))

# 将新模型中dpt部分权重字典筛选出来
dpt_part_model_dict = {}
for key, value in model_dict.items():
    if "pretrained" in key or "scratch" in key:
        dpt_part_model_dict[key] = value
print(len(dpt_part_model_dict))

#将所有权重都转移到cuda:0上
pretrained_dict_dpt_filtered_cuda = {k: v.to('cuda:0') for k, v in pretrained_dict_dpt_filtered.items()}
dpt_part_model_dict_cuda = {k: v.to('cuda:0') for k, v in dpt_part_model_dict.items()}

# 检查筛选后的dpt预训练权重keys和新模型中的dpt部分权重keys是否匹配
print(f"Are all key-value pairs match? {compare_dicts(pretrained_dict_dpt_filtered_cuda, dpt_part_model_dict_cuda)}")
print(len(pretrained_dict_dpt_filtered_cuda) == len(dpt_part_model_dict_cuda))

'''开始组合新权重文件'''
merged_weights = {**pretrained_dict_dpt_filtered, **new_model_weights_adabins_filtered}

torch.save(merged_weights, r"./weights/nyu_merged_weights.pt")