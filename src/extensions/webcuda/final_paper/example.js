function runCuda(){
	//Retrieving handle to CUDA-enabled Device
	var dev = webcuda.Device(0);
	print("Dev name: " + dev.name + " totalMem: " + dev.totalMem + " computeCapability: " + dev.computeCapability);
	//Setting up Context for CUDA Device
	var ctx = webcuda.Context(0, dev); //(flags, device handle)
	print("Ctx error: " + ctx.error + " apiVersion: "	+ ctx.apiVersion);
	//Creating host memory 
	var h_M = new Float32Array(size);
	//Creating device memory
	var d_M = webcuda.memAlloc(h_M.buffer.byteLength);
	print("d_M size: "+d_M.size+" error: "+d_M.error);
	//copying data to device
	var copyHtoD = webcuda.copyHtoD(d_M, h_M.buffer);
	print("CopyHtoD error: " + copyHtoD);
	//Loading Module
	var module = webcuda.compileFile("path/to/CUDA/code");
	print("cuName: " + module.cuName + " fname: " + module.fname + " error: " + module.error);
	//Retrieving Function from Module
	var cuFunc = webcuda.getFunction(module, "kernel_name");
	print("Func name: " + cuFunc.name + " error: " + cuFunc.error);
	//Setting kernel variables
	var gridDim = [gX,gY,gZ];
	var blockDim = [bX,bY,bZ];
	var sharedMem = 0;
	var params = [{"memParam" : d_M},
			{"intParam" : A},
			{"floatParam" : B},
			{"doubleParam" : C}];
	//Launching the Kernel
	var launchResult = webcuda.launchKernel(cuFunc, gridDim, 
			blockDim, sharedMem, params);
	print("Kernel Launch error: " + launchResult);
	//Waiting for kernel to Complete
	var sync = webcuda.synchronizeCtx();
	print("Synchronize error: " + sync);
	//Retrieving Data from CUDA Device Memory
	h_M = new Float32Array(size); 
	var copyDtoH = webcuda.copyDtoH(h_M.buffer, d_M);
	print("CopyDtoH error: " + copyDtoH);
	//Freeing CUDA Memory
	var memFree = webcuda.free(d_M);
	print("d_M free memory result: "+memFree);
	//Freeing CUDA Module
	var moduleFree = webcuda.moduleUnload(module);
	print("free module result: " + moduleFree);
	//Destroying CUDA context
	var ctxFree = webcuda.destroyCtx(ctx);
	print("free Ctx result: "+ ctxFree);
	//complete
	return "done";
}
