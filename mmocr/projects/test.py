import torch
if __name__ == '__main__':
    state_dict = torch.load('/home/user/lz/ABCNet/mmocr/projects/ABCNet/model/ic15_pretrained.pth')
    state_dict_new = state_dict['model']
    torch.save(state_dict_new,'./ic15_pre.pth')