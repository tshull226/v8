var DEBUG = 0;

var COLUMNS_BLOCKDIM_X = 16;
var COLUMNS_BLOCKDIM_Y = 8;
var COLUMNS_RESULT_STEPS = 8;
var COLUMNS_HALO_STEPS = 1;

var ROWS_BLOCKDIM_X = 16;
var ROWS_BLOCKDIM_Y = 4;
var ROWS_RESULT_STEPS = 8;
var ROWS_HALO_STEPS = 1;

//need to make input of proper typed array
function convertInput(input){
	var i;
	var size = input.length;
	var output = new Float32Array(size);
	for(i = 0; i < size; i++){
		output[i] = input[i];
	}
	return output;
}

//temp wrapper for right now
function assert(condition){
	if(condition === false){
		if(DEBUG) print("SOMETHING FAILED");
	}
}

function runCUDA(h_K, h_I){
	//Retrieving Device
	if(DEBUG) print("retrieving Device Info");
	profiler.start("Retrieving device info");
	var dev = webcuda.Device(0);
	profiler.stop("Retrieving device info");

	//Setting up Context for CUDA Device
	profiler.start("Creating CUDA context");
	if(DEBUG) print("creating Context");
	var ctx = webcuda.Context(0, dev);
	profiler.stop("Creating CUDA context");

	//Creating host memory for pixel array
	if(DEBUG) print("creating host memory");
	profiler.start("Allocating host memory");
	var h_Kernel = convertInput(h_K);
	var h_Input = convertInput(h_I);
	var h_Output = new Float32Array(imageW * imageH);
	var h_Buffer = new Float32Array(imageW * imageH);
	profiler.stop("Allocating host memory");

	//Creating device memory for pixel array
	if(DEBUG) print("allocating CUDA memory");
	profiler.start("Allocating CUDA memory");
	var d_Input = webcuda.memAlloc(h_Input.buffer.byteLength);
	var d_Output = webcuda.memAlloc(h_Output.buffer.byteLength);
	var d_Buffer = webcuda.memAlloc(h_Buffer.buffer.byteLength);
	var d_Kernel = webcuda.memAlloc(h_Kernel.buffer.byteLength);
	if(DEBUG) print("d_Input size: "+d_Input.size+" error: "+d_Input.error);
	if(DEBUG) print("d_A size: "+d_Output.size+" error: "+d_Output.error);
	if(DEBUG) print("d_A size: "+d_Buffer.size+" error: "+d_Buffer.error);
	profiler.stop("Allocating CUDA memory");


	//copying data to device
	if(DEBUG) print("copying CUDA initial parameters to device");
	profiler.start("copyHtoD");
	//TODO should allow them to be asynchronous
	var memCpy = webcuda.copyHtoD(d_Input, h_Input.buffer);
	if(DEBUG) print("memCopy result: " + memCpy);
	var memCpy = webcuda.copyHtoD(d_Kernel, h_Kernel.buffer);
	if(DEBUG) print("memCopy result: " + memCpy);
	profiler.stop("copyHtoD");

	//need the convolutionKernel function
	//or maybe just add as parameter...
	
	//Compiling Module
	if(DEBUG) print("compiling CUDA module");
	profiler.start("Loading CUDA module");
	var module = webcuda.compileFile("tests/image_convolution/WebCUDA/convolutionSeparable");
	if(DEBUG) print("cuName: " + module.cuName + " fname: " + module.fname + " error: " + module.error);
	profiler.stop("Loading CUDA module");

	//Retrieving Function from Module
	if(DEBUG) print("retrieving function from module");
	profiler.start("Retrieving function from module");
	var rowFunc = webcuda.getFunction(module, "convolutionRowsKernel");
	if(DEBUG) print("name: " + rowFunc.name + " error: " + rowFunc.error);
	var columnFunc = webcuda.getFunction(module, "convolutionColumnsKernel");
	if(DEBUG) print("name: " + columnFunc.name + " error: " + columnFunc.error);
	profiler.stop("Retrieving function from module");

	//looping through kernel
	profiler.start("kernel");
	for(var i = 0; i < numIterations; i++){
		convolutionRowsGPU(rowFunc, d_Buffer, d_Input, d_Kernel, imageW, imageH);
		webcuda.synchronizeCtx();
		convolutionColumnsGPU(columnFunc, d_Output, d_Buffer, d_Kernel, imageW, imageH);
		webcuda.synchronizeCtx();
	}
	profiler.stop("kernel");
	
	//Retrieving Data from CUDA Device Memory
	if(DEBUG) print("copying CUDA Mem Result to device");
	profiler.start("copyDtoH");
	//TODO should allow them to be asynchronous
	var copyDtoH = webcuda.copyDtoH(h_Output.buffer, d_Output);
	if(DEBUG) print("copying result: " + copyDtoH);
	profiler.stop("copyDtoH");

	//Freeing CUDA Memory
	if(DEBUG) print("freeing CUDA memory");
	profiler.start("Freeing CUDA memory");
	var memFree = webcuda.free(d_Input);
	if(DEBUG) print("d_Input free memory result: "+memFree);
	var memFree = webcuda.free(d_Output);
	if(DEBUG) print("d_Output free memory result: "+memFree);
	var memFree = webcuda.free(d_Buffer);
	if(DEBUG) print("d_Buffer free memory result: "+memFree);
	profiler.stop("Freeing CUDA memory");

	//Freeing CUDA Module
	profiler.start("Freeing CUDA module");
	if(DEBUG) print("freeing CUDA module");
	var moduleFree = webcuda.moduleUnload(module);
	if(DEBUG) print("free module result: " + moduleFree);
	profiler.stop("Freeing CUDA module");

	//Destroying CUDA context
	if(DEBUG) print("destroying CUDA context");
	profiler.start("Destorying CUDA context");
	var ctxFree = webcuda.destroyCtx(ctx);
	if(DEBUG) print("free context result: "+ ctxFree);
	profiler.stop("Destorying CUDA context");

	return d_Output;

}

function convolutionRowsGPU(
		func,
		d_Dst,
		d_Src,
		d_Kernel,
		imageW,
		imageH
		)
{
	assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
	assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
	assert(imageH % ROWS_BLOCKDIM_Y == 0);

	//dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
	//dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);
	var blocks = [imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y, 1];
	var threads = [ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y, 1];

	/*
		 convolutionRowsKernel<<<blocks, threads>>>(
		 d_Dst,
		 d_Src,
		 imageW,
		 imageH,
		 imageW
		 );
		 getLastCudaError("convolutionRowsKernel() execution failed\n");
		 */
	var launchResult = webcuda.launchKernel(func, blocks, threads, 0, 
			[{"memParam" : d_Dst},
			{"memParam" : d_Src},
			{"memParam" : d_Kernel},
			{"intParam" : imageW},
			{"intParam" : imageH},
			{"intParam" : imageW}]);
	if(DEBUG) print("launch Result: " + launchResult);
}

function convolutionColumnsGPU(
		func,
		d_Dst,
		d_Src,
		d_Kernel,
		imageW,
		imageH
		)
{
	assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
	assert(imageW % COLUMNS_BLOCKDIM_X == 0);
	assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

	//dim3 blocks(imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
	//dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
	var blocks = [imageW / COLUMNS_BLOCKDIM_X, imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y), 1];
	var threads = [COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y, 1];

	/*
		 convolutionColumnsKernel<<<blocks, threads>>>(
		 d_Dst,
		 d_Src,
		 imageW,
		 imageH,
		 imageW
		 );
		 getLastCudaError("convolutionColumnsKernel() execution failed\n");
		 */

	var launchResult = webcuda.launchKernel(func, blocks, threads, 0, 
			[{"memParam" : d_Dst},
			{"memParam" : d_Src},
			{"memParam" : d_Kernel},
			{"intParam" : imageW},
			{"intParam" : imageH},
			{"intParam" : imageW}]);
	if(DEBUG) print("launch Result: " + launchResult);
}
