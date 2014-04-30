function bs_webcuda(h_StockPrice, h_OptionStrike, h_OptionYears, RiskFree, Volatility, optN) {
	//Retrieving Device
	print("retrieving Device Info");
	var dev = webcuda.Device(0);

	//Setting up Context for CUDA Device
	print("creating Context");
	var ctx = webcuda.Context(0, dev);

	//Creating host memory for pixel array
	print("creating host memory");
	var h_CallResult = new Float32Array(optN); 
	var h_PutResult = new Float32Array(optN); 

	//Creating device memory for pixel array
	print("allocating CUDA memory");
	var d_CallResult = webcuda.memAlloc(h_CallResult.buffer.byteLength);
	var d_PutResult = webcuda.memAlloc(h_PutResult.buffer.byteLength);
	var d_StockPrice = webcuda.memAlloc(h_StockPrice.buffer.byteLength);
	var d_OptionStrike = webcuda.memAlloc(h_OptionStrike.buffer.byteLength);
	var d_OptionYears = webcuda.memAlloc(h_OptionYears.buffer.byteLength);
	print("error: "+d_CallResult.error);
	print("error: "+d_PutResult.error);
	print("error: "+d_StockPrice.error);
	print("error: "+d_OptionStrike.error);
	print("error: "+d_OptionYears.error);
	//print("size: "+d_I.size+" error: "+d_I.error);

	//Sending Input Data to CUDA Device Memory
	print("copying input data to device");
	webcuda.copyHtoD(d_StockPrice, h_StockPrice.buffer);
	webcuda.copyHtoD(d_OptionStrike, h_OptionStrike.buffer);
	webcuda.copyHtoD(d_OptionYears, h_OptionYears.buffer);

	//Loading Module
	print("loading CUDA module");
	//var module = webcuda.compileFile("tests/blackscholes/WebCUDA/bs_kernel");
	var module = webcuda.moduleLoad("tests/blackscholes/bs_kernel.ptx");
	print("fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "BlackScholesGPU");
	print("name: " + cuFunc.name + " error: " + cuFunc.error);

	//Launching the Kernel
	print("trying to launch kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [optN/128,1,1], [128,1,1], 0,
						[ {"memParam" : d_CallResult}, {"memParam" : d_PutResult},
						  {"memParam" : d_StockPrice}, {"memParam" : d_OptionStrike}, {"memParam" : d_OptionYears},
						  {"floatParam" : RiskFree}, {"floatParam" : Volatility}, {"intParam" : optN}
						]);
	print("launch result: " + launchResult);

	//Retrieving Data from CUDA Device Memory
	print("copying CUDA Mem Result to host");
	webcuda.copyDtoH(h_CallResult.buffer, d_CallResult);
	webcuda.copyDtoH(h_PutResult.buffer, d_PutResult);

	/*
	//temp check to see if things seem reasonable
	print("checking results");
	for(i = 0; i < numPixels; i++){
	print(h_I[i]);
	}
	*/

	//Freeing CUDA Memory
	print("freeing CUDA memory");
	var memFree = webcuda.free(d_CallResult);
	print("free memory result: " + memFree);
	var memFree = webcuda.free(d_PutResult);
	print("free memory result: " + memFree);
	var memFree = webcuda.free(d_StockPrice);
	print("free memory result: " + memFree);
	var memFree = webcuda.free(d_OptionStrike);
	print("free memory result: " + memFree);
	var memFree = webcuda.free(d_OptionYears);
	print("free memory result: " + memFree);

	//Freeing CUDA Module
	print("freeing CUDA module");
	var moduleFree = webcuda.moduleUnload(module);
	print("free module result: " + moduleFree);

	//Destroying CUDA context
	print("destroying CUDA context");
	var ctxFree = webcuda.destroyCtx(ctx);
	print("free context result: "+ ctxFree);

	//returning value
	return [h_CallResult, h_PutResult];
}
