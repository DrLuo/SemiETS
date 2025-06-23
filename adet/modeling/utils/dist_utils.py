import torch
import torch.distributed as dist
from detectron2.utils import comm


@torch.no_grad()
def concat_all_gather_with_various_shape(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # current_rank = torch.distributed.get_rank()
    # print('current_rank: ', current_rank)
    # if len(tensor.size()) == 1:
    #     tensor = tensor.view(-1, 1)

    # tensor_size = torch.tensor(tensor.size()).to(tensor.device)
    # device = tensor.device
    # dtype = tensor.dtype

    # size_gather = [torch.zeros_like(tensor_size) for _ in range(torch.distributed.get_world_size())]
    # torch.distributed.all_gather(size_gather, tensor_size, async_op=False)
    # tensors_gather = [torch.zeros(torch.Size(_size), dtype=dtype).to(device) for _size in size_gather]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    tensors_gather = comm.all_gather(tensor)
    tensors_gather = [ t.detach().cpu() for t in tensors_gather]

    # tensors_gather.pop(current_rank)
    # print('>>>>> Right local?',(tensor == tensors_gather[current_rank]).all())
    tensors_gather = torch.cat(tensors_gather)

    # output = torch.cat(tensors_gather, dim=0)
    return tensors_gather#, current_rank


@torch.no_grad()
def concat_all_gather(tensor):
    # gather all tensor shape
    shape_tensor = torch.tensor(tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(comm.get_world_size())]
    comm.all_gather(shape_list, shape_tensor)

    # padding tensor to the max length
    if shape_list[0].numel() > 1:
        max_shape = torch.tensor([_[0] for _ in shape_list]).max()
        padding_tensor = torch.zeros((max_shape, shape_tensor[1]), device='cuda').type_as(tensor)
    else:
        max_shape = torch.tensor(shape_list).max()
        padding_tensor = torch.zeros(max_shape, device='cuda').type_as(tensor)

    padding_tensor[:shape_tensor[0]] = tensor

    tensor_list = [torch.zeros_like(padding_tensor) for _ in range(comm.get_world_size())]
    comm.all_gather(tensor_list, padding_tensor)

    sub_tensor_list = []
    for sub_tensor, sub_shape in zip(tensor_list, shape_list):
        sub_tensor_list.append(sub_tensor[:sub_shape[0]])
    output = torch.cat(sub_tensor_list, dim=0)

    return output


@torch.no_grad()
def concat_all_gather_equal_size(tensor, dim=0):
    """Performs all_gather operation on the provided tensors.

    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=dim)
    return output
