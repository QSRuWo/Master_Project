import torch


class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.

        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

# # 定义BaseModel类，该类继承了PyTorch的torch.nn.Module类。
# class BaseModel(torch.nn.Module):
#
#     # 定义一个名为load的方法，用于从文件加载模型。
#     def load(self, path):
#         """Load model from file.
#
#         Args:
#             path (str): file path
#         """
#
#         # 使用torch.load函数从指定的文件路径加载模型参数。
#         # map_location参数确保参数被加载到CPU上。
#         parameters = torch.load(path, map_location=torch.device("cpu"))
#
#         # 检查加载的参数字典中是否有名为"optimizer"的键。
#         # 如果存在，说明保存的是包含模型参数和优化器参数的字典。
#         # 在这种情况下，我们只提取模型参数。
#         if "optimizer" in parameters:
#             parameters = parameters["model"]
#
#         # 使用load_state_dict方法加载模型参数。
#         self.load_state_dict(parameters)
