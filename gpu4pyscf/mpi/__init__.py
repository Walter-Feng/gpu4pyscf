from mpi4py import MPI
import cupy
import cupy.cuda.nccl as nccl

def to_nccl_data_type(cupy_array):

    nccl_type_dict = {
        "ncclInt8"       : 0, "ncclChar"       : 0,
        "ncclUint8"      : 1,
        "ncclInt32"      : 2, "ncclInt"        : 2,
        "ncclUint32"     : 3,
        "ncclInt64"      : 4,
        "ncclUint64"     : 5,
        "ncclFloat16"    : 6, "ncclHalf"       : 6,
        "ncclFloat32"    : 7, "ncclFloat"      : 7,
        "ncclFloat64"    : 8, "ncclDouble"     : 8,
    }
    
    return nccl_type_dict["nccl" + str(cupy_array.dtype).capitalize()]


class Communicator:
    def __init__(self):
    
        self.world = MPI.COMM_WORLD

        self.is_main = (self.world.rank == 0)

        unique_id = nccl.get_unique_id()
        unique_id = self.world.bcast(unique_id)

        processor_name = MPI.Get_processor_name()
        rank = self.world.rank

        host_names = self.world.gather(processor_name)


        # This removes redundant host names. Also the order can be random
        # if the removal is operated individually
        if self.is_main:
            host_names = list(set(host_names))

        host_names = self.world.bcast(host_names)
        color = host_names.index(processor_name)

        n_gpu = cupy.cuda.runtime.getDeviceCount()

        self.local = self.world.Split(color, rank)
        local_rank = self.local.rank
        cupy.cuda.Device(local_rank).use()

        if self.local.size > n_gpu:
            raise Exception("the size of local processes exceeds allocable GPU devices")

        self.gpu = nccl.NcclCommunicator(self.world.size, unique_id, rank)

    def reduce_on_gpu(self, cupy_array : cupy.ndarray):
        nccl_sum_type = 0
        default_stream = 0
        self.gpu.allReduce(cupy_array.data.ptr, cupy_array.data.ptr, 
                           cupy_array.size, to_nccl_data_type(cupy_array), 
                           nccl_sum_type, default_stream)

