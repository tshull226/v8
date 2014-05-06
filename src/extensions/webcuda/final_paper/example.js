% {\normalsize \bf // Context Creation} *)
//Retrieving handle to CUDA-enabled Device
var dev = webcuda.Device(0);
print("Dev name: " + dev.name + " totalMem: " + dev.totalMem 
		+ " computeCapability: " + dev.computeCapability);
//Setting up Context for CUDA Device
var ctx = webcuda.Context(flags, dev); 
% {\normalsize \bf // Memory Allocation} *)
var h_Mem = new Float32Array(size); //Creating host memory 
var d_Mem = webcuda.memAlloc(h_Mem.buffer.byteLength); //Creating device memory
% {\normalsize \bf // Data Communication} *)
var copyHtoD = webcuda.copyHtoD(d_Mem, h_Mem.buffer); //copying data to device
% {\normalsize \bf // Kernel Setup} *)
var module = webcuda.compileFile("path/to/CUDA/code"); //Loading Module
//Retrieving Function from Module
var cuFunc = webcuda.getFunction(module, "kernel_name"); 
% {\normalsize \bf // Kernel Execution} *)
//Setting kernel variables
var gridDim = [gX,gY,gZ];
var blockDim = [bX,bY,bZ];
var sharedMem = 0;
var params = [{"memParam" : d_Mem},
	{"intParam" : A},
	{"floatParam" : B},
	{"doubleParam" : C}];
//Launching the Kernel
var launchResult = webcuda.launchKernel(cuFunc, gridDim, blockDim, sharedMem, params);
var sync = webcuda.synchronizeCtx(); //Waiting for kernel to Complete
% {\normalsize \bf // Data Communication} *)
//Retrieving Data from CUDA Device Memory
h_Mem = new Float32Array(size);  
var copyDtoH = webcuda.copyDtoH(h_Mem.buffer, d_Mem);
% {\normalsize \bf // Freeing Resources} *)
var memFree = webcuda.free(d_Mem); //Freeing CUDA Memory
var moduleFree = webcuda.moduleUnload(module); //Freeing CUDA Module
var ctxFree = webcuda.destroyCtx(ctx); //Destroying CUDA context
return "done"; //complete
