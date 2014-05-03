var DEBUG = 0;

function bs_webcuda(h_StockPrice, h_OptionStrike, h_OptionYears, RiskFree, Volatility, optN, profiler, printDebug) {
	if(printDebug != undefined) DEBUG = 1;

	//Retrieving Device
	if(DEBUG) print("retrieving Device Info");
	profiler.start("Retrieving device info");
	var dev = webcuda.Device(0);
	profiler.stop("Retrieving device info");

	//Setting up Context for CUDA Device
	if(DEBUG) print("creating CUDA context");
	profiler.start("Creating CUDA context");
	var ctx = webcuda.Context(0, dev);
	profiler.stop("Creating CUDA context");

	//Creating host memory for pixel array
	if(DEBUG) print("creating host memory");
	profiler.start("Allocating host memory");
	var h_CallResult = new Float32Array(optN); 
	var h_PutResult = new Float32Array(optN); 
	profiler.stop("Allocating host memory");

	//Creating device memory for pixel array
	if(DEBUG) print("allocating CUDA memory");
	profiler.start("Allocating CUDA memory");
	var d_CallResult = webcuda.memAlloc(h_CallResult.buffer.byteLength);
	var d_PutResult = webcuda.memAlloc(h_PutResult.buffer.byteLength);
	var d_StockPrice = webcuda.memAlloc(h_StockPrice.buffer.byteLength);
	var d_OptionStrike = webcuda.memAlloc(h_OptionStrike.buffer.byteLength);
	var d_OptionYears = webcuda.memAlloc(h_OptionYears.buffer.byteLength);
	//if(DEBUG) print("size: "+d_I.size+" error: "+d_I.error);
	profiler.stop("Allocating CUDA memory");

	//Sending Input Data to CUDA Device Memory
	if(DEBUG) print("copying input data to device");
	profiler.start("copyHtoD");
	webcuda.copyHtoD(d_StockPrice, h_StockPrice.buffer);
	webcuda.copyHtoD(d_OptionStrike, h_OptionStrike.buffer);
	webcuda.copyHtoD(d_OptionYears, h_OptionYears.buffer);
	profiler.stop("copyHtoD");

	//Loading Module
	if(DEBUG) print("loading CUDA module");
	profiler.start("Loading CUDA module");
	var module = webcuda.compileFile("tests/blackscholes/WebCUDA/bs_kernel");
	//var module = webcuda.moduleLoad("tests/blackscholes/WebCUDA/bs_kernel.ptx");
	if(DEBUG) print("fname: " + module.fname + " error: " + module.error);
	profiler.stop("Loading CUDA module");

	//Retrieving Function from Module
	if(DEBUG) print("retrieving function from module");
	profiler.start("Retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "BlackScholesGPU");
	if(DEBUG) print("name: " + cuFunc.name + " error: " + cuFunc.error);
	profiler.stop("Retrieving function from module");

	//Launching the Kernel
	if(DEBUG) print("trying to launch kernel");
	profiler.start("kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [optN/128,1,1], [128,1,1], 0,
						[ {"memParam" : d_CallResult}, {"memParam" : d_PutResult},
						  {"memParam" : d_StockPrice}, {"memParam" : d_OptionStrike}, {"memParam" : d_OptionYears},
						  {"floatParam" : RiskFree}, {"floatParam" : Volatility}, {"intParam" : optN}
						]);
	if(DEBUG) print("launch result: " + launchResult);
	profiler.stop("kernel");

	//Retrieving Data from CUDA Device Memory
	if(DEBUG) print("copying CUDA Mem Result to host");
	profiler.start("copyDtoH");
	webcuda.copyDtoH(h_CallResult.buffer, d_CallResult);
	webcuda.copyDtoH(h_PutResult.buffer, d_PutResult);
	profiler.stop("copyDtoH");

	/*
	//temp check to see if things seem reasonable
	if(DEBUG) print("checking results");
	for(i = 0; i < numPixels; i++){
	if(DEBUG) print(h_I[i]);
	}
	*/

	//Freeing CUDA Memory
	if(DEBUG) print("freeing CUDA memory");
	profiler.start("Freeing CUDA memory");
	var memFree = webcuda.free(d_CallResult);
	if(DEBUG) print("free memory result: " + memFree);
	var memFree = webcuda.free(d_PutResult);
	if(DEBUG) print("free memory result: " + memFree);
	var memFree = webcuda.free(d_StockPrice);
	if(DEBUG) print("free memory result: " + memFree);
	var memFree = webcuda.free(d_OptionStrike);
	if(DEBUG) print("free memory result: " + memFree);
	var memFree = webcuda.free(d_OptionYears);
	if(DEBUG) print("free memory result: " + memFree);
	profiler.stop("Freeing CUDA memory");

	//Freeing CUDA Module
	if(DEBUG) print("freeing CUDA module");
	profiler.start("Freeing CUDA module");
	var moduleFree = webcuda.moduleUnload(module);
	if(DEBUG) print("free module result: " + moduleFree);
	profiler.stop("Freeing CUDA module");

	//Destroying CUDA context
	if(DEBUG) print("destroying CUDA context");
	profiler.start("Destorying CUDA context");
	var ctxFree = webcuda.destroyCtx(ctx);
	if(DEBUG) print("free context result: "+ ctxFree);
	profiler.stop("Destorying CUDA context");

	//returning value
	return [h_CallResult, h_PutResult];
}
