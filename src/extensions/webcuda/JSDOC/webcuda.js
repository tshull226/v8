/**
 * @fileOverview WebCUDA JavaScript API Documentation
 * @author Jiho Choi, Kurt Fellows, Tom Shull
 */


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
	 * @return 
	 */
	"stopProfiling" : function(){},
	/**
	 * creates a handle to the specified CUDA device.
	 * @param {int} deviceNum the number of CUDA-device to attach to
	 * @return {Device} a new {@link Device} Object. 	 
	 */
	"Device" : function(deviceNum){},
	/**
	 * @deprecated
	 * not sure why this is here
	 */
	"Module" : function(){},
	/**
	 * Loads .ptx or .cubin from given location into a handle for a CUDA module 
	 * @param {string} filename the location of file to be loaded
	 * @return {Module} a new {@link Module} Object
	 */
	"moduleLoad" : function(fileName){},
	/**
	 * Extracts requested kernel from the CUDA module and loads it into a CUDA function handle
	 * @param {Module} module the CUDA module to load the kernel from
	 * @param {string} kernelName the name of the kernel
	 * @return {Kernel} a new {@link Kernel} Object
	 */
	"getFunction" : function(module, kernelName){},
	/**
	 * compiles the given string into a handle to a CUDA module
	 * @param {string} module
	 * @return {Module} a new {@link Module} Object
	 */
	"compile" : function(module){},
	/**
	 * compile a text file into a handle to a CUDA module
	 * @param {string} fileName the file path to text file
	 * @return {Module} a new {@link Module} Object
	 */
	"compileFile" : function(fileName){},
	/**
	 * allocates memory on CUDA device for use in kernels
	 * @param {int} bytes the amount of bytes to be allocated on device. NOTE: the device memory will be allocated on the device that is associated with the current context. This can be retrieved via calling {@link webcuda.getCurrent}.
	 * @return {Mem} a new {@link Mem} object
	 */
	"memAlloc" : function(bytes){},
	/**
	 * @deprecated (NEED TO ADJUST)
	 * allocates memory on CUDA device for use in kernels
	 * @param {int} bytes the amount of bytes to be allocated on device. NOTE: the device memory will be allocated on the device that is associated with the current context. This can be retrieved via calling {@link webcuda.getCurrent}.
	 * @return {Mem} a new {@link Mem} object
	 */
	"memAllocPitch" : function(){},
	/**
	 * frees CUDA device memory
	 * @param {Mem} deviceMem the CUDA device memory to be freed
	 * @return {int} the error number denoting whether the operation was successful (0) or not (!0) 
	 */
	"free" : function(deviceMem){},
	/**
	 *
	 */
	"copyHtoD" : function(hostMem, deviceMem){},
	"copyDtoH" : function(){},
	"newContext" : function(){},
	"synchronize" : function(){},
	"setCurrent" : function(){},
	/**
	 * Returns the a handle to the current CUDA context
	 * @return {Context} a {@link Context} Object holding the current context
	 */
	"getCurrent" : function(){},
	"pushCurrent" : function(){},
	"popCurrent" : function(){},
	"launchKernel" : function(){},
	/**
	 * @deprecated Need to see what this does
	 */
	"Function" : function(){},
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
	 *
	 */
	this.name = 0;
	/**
	 *
	 */
	this.error = 0;
}

/** @class */
function Kernel() {
	/**
	 *
	 */
	this.name = 0;
	/**
	 *
	 */
	this.error = 0;
}
