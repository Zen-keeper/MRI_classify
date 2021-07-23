import  os
#简单使用
from pynvml import *
def set_gpu():
    nvmlInit()     #初始化
    print("Driver: ",nvmlSystemGetDriverVersion())
    #显示驱动信息
    #>>> Driver: 384.xxx
    free_gpu = ""
    #查看设备
    deviceCount = nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        print("GPU", i, ":", nvmlDeviceGetName(handle))
        #查看显存、温度、风扇、电源
        # info = nvmlDeviceGetMemoryInfo(handle)
        # print("Memory Total: ",info.total)
        # print("Memory Free: ",info.free)
        # print("Memory Used: ",info.used)

        print("Temperature is %d C"%nvmlDeviceGetTemperature(handle,0))
        print("Fan speed is ", nvmlDeviceGetFanSpeed(handle))
        print("Power ststus",nvmlDeviceGetPowerState(handle))
        # if (info.free >1000*1024*1024 ):
        #     if(not free_gpu==""):
        # free_gpu = free_gpu + ","
        # free_gpu=free_gpu+(str(i))
    #最后要关闭管理工具
    nvmlShutdown()
    # print(free_gpu)

    #nvmlDeviceXXX有一系列函数可以调用，包括了NVML的大多数函数。
    #具体可以参考：https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html#group__nvmlDeviceQueries
    # if(len(free_gpu)<2):
    free_gpu = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = free_gpu
    print(os.environ["CUDA_VISIBLE_DEVICES"] )

if __name__=='__main__':
    set_gpu()