
/** @namespace*/
var webcuda = {
	/**
	 * method to query the version of WebCUDA being used
	 * @return {string} version number of WebCUDA
	 */
	"version" : function(){},
	/**
	 * method to query current Driver Versions of the CUDA client
	 * @return {int} driver version number of WebCUDA
	 */
	"driverVersion" : function(){},
	/**
	 * method to query the number of CUDA devices on the machine
	 * @return {int} number of CUDA-enabled devices attached 
	 */
	"deviceCount" : function(){},
	/**
	 * notifies NVidia Visual Profiler to begin profiling
	 */
	"startProfiling" : function(){},
	/**
	 * notifies Nvidia Visual Profiler to end profiling
	 */
	"stopProfiling" : function(){},
	/**
	 * creates a handle to the specified CUDA device.
	 * @param {int} deviceNum the number of CUDA-device to attach to
	 * @return {Device} a new {@link Device} Object. 	 
	 */
	"Device" : function(deviceNum){},
	/**
	 * Loads .ptx or .cubin from given location into a handle for a CUDA module 
	 * @param {string} filename the location of file to be loaded
	 * @return {Module} a new {@link Module} Object
	 */
	"moduleLoad" : function(fileName){},
	/**
	 * unload  a cuModule and delete any temporary files created
	 * @param {Module} handle for the module to be unloaded
	 * @return {int} the error number denoting whether the operation was successful (0) or not (not 0) 
	 */
	"moduleUnload" : function(module){},
	/**
	 * Extracts requested kernel from the CUDA module and loads it into a CUDA function handle
	 * @param {Module} module the CUDA module to load the kernel from
	 * @param {string} kernelName the name of the kernel
	 * @return {Kernel} a new {@link Kernel} Object
	 */
	"getFunction" : function(module, kernelName){},
	/**
	 * @deprecated need to create new module type
	 * compiles the given string into a handle to a CUDA module
	 * @param {string} module
	 * @return {ModuleStr} a new {@link Module} Object
	 */
	"compile" : function(module){},
	/**
	 * compile a cu file into a handle to a CUDA module
	 * @param {string} fileName (without .cu) for the file path to cu file
	 * @return {ModuleTxt} a new {@link Module} Object
	 */
	"compileFile" : function(fileName){},
	/**
	 * compile a txt file into a handle to a CUDA module
	 * @param {string} fileName (without .txt) for the file path to text file
	 * @return {ModuleTxt} a new {@link Module} Object
	 */
	"compileText" : function(fileName){},
	/**
	 * allocates memory on CUDA device for use in kernels. NOTE: the device memory will be allocated on the device that is associated with the current context. This can be retrieved via calling {@link webcuda.getCurrent}.
	 * @param {int} bytes the amount of bytes to be allocated on device. 
	 * @return {Mem} a new {@link Mem} object
	 */
	"memAlloc" : function(bytes){},
	/**
	 * allocates memory on CUDA device for use in kernels. NOTE: the device memory will be allocated on the device that is associated with the current context. This can be retrieved via calling {@link webcuda.getCurrent}.
	 * @param {int} width the allocation width in bytes
	 * @param {int} height the allocation height in bytes
	 * @param {int} bytes maximum byte size of any value to be allocated (can be either 4, 8, or 16)
	 * @return {MemPitch} a new {@link MemPitch} object
	 */
	"memAllocPitch" : function(width, height, bytes){},
	/**
	 * frees CUDA device memory
	 * @param {Mem} deviceMem the CUDA device memory to be freed
	 * @return {int} the error number denoting whether the operation was successful (0) or not (not 0) 
	 */
	"free" : function(deviceMem){},
	/**
	 * Copies memory from JavaScript Array into CUDA Device Memory
	 * @param {Mem} deviceMem CUDA Device memory to copy values into
	 * @param {int[]|float[]|double[]} hostMem JavaScript Array to copy host memory into
	 * @return {Number} the error number denoting the operation was successful (0) or not (not 0)
	 */
	"copyHtoD" : function(deviceMem, hostMem){},
	/**
	 * Copies memory from CUDA device into JavaScript Array
	 * @param {int[]|float[]|double[]} hostMem JavaScript Array to copy host memory into
	 * @param {Mem} deviceMem CUDA Device memory to copy values into
	 * @return {Number} the error number denoting the operation was successful (error==0) or not (error!=0)
	 */
	"copyDtoH" : function(hostMem, deviceMem){},
	/**
	 * creates a new CUDA context for the specified device
	 * @param {int} flags The various settings for a context. See <a href="http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__CTX_g65dc0012348bc84810e2103a40d8e2cf.html">cuCtxCreate</a> for more details
	 * @param {Device} device
	 * @return {Context} a new {@link Context} object
	 */
	"Context" : function(flags, device){},
	/**
	 * stall host thread until all CUDA device operations are complete
	 * @return {Number} the error number denoting whether the operation was successful (error==0) or not (error !=0)
	 */
	"synchronizeCtx" : function(){},
	/**
	 * changes CUDA context to input and pops the current context (if any)
	 * @param {Context} ctx the CUDA context to make the current context
	 * @return {Number} the error number denoting whether the operation was successful (error==0) or not (error !=0)
	 */
	"setCurrentCtx" : function(ctx){},
	/**
	 * Returns the a handle to the current CUDA context
	 * @return {Context} a {@link Context} Object holding the current context
	 */
	"getCurrentCtx" : function(){},
	/**
	 * changes current CUDA context to input. Like {@link webcuda.setCurrent} except pushes current context onto the stack
	 * @param {Context} ctx the CUDA context to make the current context
	 * @return {Number} the error number denoting whether the operation was successful (error==0) or not (error !=0)
	 */
	"pushCurrentCtx" : function(ctx){},
	/**
	 * pops the current CUDA context and restores the previous context in the stack (if any)
	 * @return {Context} the current {@link Context} the was removed from the stack
	 */
	"popCurrentCtx" : function(){},
	/**
	 * Destroys specified context and pops it if local to the calling thread. Please see <a href="http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__CTX_g27a365aebb0eb548166309f58a1e8b8e.html#g27a365aebb0eb548166309f58a1e8b8e">cuCtxDestroy</a> for more details
	 * @param {Context} ctx the CUDA context to destroy
	 * @return {Number} the error number denoting whether the operation was successful (error==0) or not (error !=0)
	 */
	"destroyCtx" : function(ctx){},
	/**
	 * asynchronous launches the cuda kernel specified. See <a href="http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/docs/online/group__CUDA__EXEC_gb8f3dc3031b40da29d5f9a7139e52e15.html#gb8f3dc3031b40da29d5f9a7139e52e15">cuLaunchKernel</a> for more details
	 * @param {int[]} gridDim the grid dimensions (3D) for launch. NOTE: to work correctly must have an array of length 3
	 * @param {int[]} blockDim the threadblock dimensions (3D) for launch. NOTE: to work correctly must have an array of length 3
	 * @param {int} the number of bytes of shared memory per thread block
	 * @param {Object[]} kernelArgs the arguments for the kernel to be launched. Each element in the array must an object with a the property name of either "memParam", "intParam", "floatParam", or "doubleParam" and have the corresponding value with the description
	 * @return {Number} the error number denoting whether the operation was successful (error==0) or not (error !=0)
	 */
	"launchKernel" : function(gridDim, blockDim, sharedMem, kernelArgs){},
};

/** @class */
function Device() {
	/**
	 *
	 */
	this.name = 0;
	/**
	 *
	 */
	this.major = 0;
	/**
	 *
	 */
	this.minor = 0;
}

/** @class */
function Module() {
	/**
	 * {string} name of the file kernel was extracted from
	 */
	this.fname = 0;
	/**
	 * error value deonting whether the loading of the module was successful (error==0) or not (error!=0)
	 */
	this.error = 0;
}

/** @class */
function Kernel() {
	/**
	 * {string} name of the of the kernel
	 */
	this.name = 0;
	/**
	 * error value denoting whether retrieval of the kernel was successful (error==0) or not (error!=0)
	 */
	this.error = 0;
}

/** @class */
function Mem() {
	/**
	 * the number of bytes allocated
	 */
	this.size = 0;
	/**
	 * error value denoting whether allocation was successful (error==0) or not (error!=0)
	 */
	this.error = 0;
}

/** @class */
function MemPitch() {
	/**
	 * the number of elements in row
	 */
	this.width = 0;
	/**
	 * the number of elements in column
	 */
	this.height = 0;
	/**
	 * the number of bytes per element
	 */
	this.pitch = 0;
	/**
	 * error value denoting whether allocation was successful (error==0) or not (error!=0)
	 */
	this.error = 0;
}

/** @class */
function Context() {
	/**
	 * {Number} value denoting whether the context was created successfully (error==0) or not (error!=0)
	 */
	this.error = 0;
	/**
	 * {Number} the Api Version of the thread used to create the context
	 */
	this.ApiVersion = 0;
}
