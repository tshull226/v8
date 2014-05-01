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
	for(i = 0; i < size, i++){
		output[i] = input[i];
	}
	return output;
}

//temp wrapper for right now
function assert(condition){
	if(condition === false){
		print("SOMETHING FAILED");
	}
}

function runCUDA(h_K, h_I){

	//Retrieving Device
	print("retrieving Device Info");
	var dev = webcuda.Device(0);

	//Setting up Context for CUDA Device
	print("creating Context");
	var ctx = webcuda.Context(0, dev);

	//Creating host memory for pixel array
	print("creating host memory");
	var h_Kernel = convertInput(h_K);
	var h_Input = convertInput(h_I);
	var h_Output = new Float32Array(imageW * imageH);
	var h_Buffer = new Float32Array(imageW * imageH);

	//Creating device memory for pixel array
	print("allocating CUDA memory");
	var d_Input = webcuda.memAlloc(h_Input.buffer.byteLength);
	var d_Output = webcuda.memAlloc(h_Output.buffer.byteLength);
	var d_Buffer = webcuda.memAlloc(h_Buffer.buffer.byteLength);
	print("d_Input size: "+d_Input.size+" error: "+d_Input.error);
	print("d_A size: "+d_Output.size+" error: "+d_Output.error);
	print("d_A size: "+d_Buffer.size+" error: "+d_Buffer.error);


	//copying data to device
	print("copying CUDA initial parameters to device");
	//TODO should allow them to be asynchronous
	var memCpy = webcuda.copyHtoD(d_X, h_X.buffer);
	print("memCopy result: " + memCpy);

	//need the convolutionKernel function
	//or maybe just add as parameter...
	
	//Compiling Module
	print("compiling CUDA module");
	var module = webcuda.compileFile("tests/image_convolution/WebCUDA/convolutionSeparable");
	print("cuName: " + module.cuName + " fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	var rowFunc = webcuda.getFunction(module, "convolutionRowsKernel");
	print("name: " + rowFunc.name + " error: " + rowFunc.error);
	var columnFunc = webcuda.getFunction(module, "convolutionColumnsKernel");
	print("name: " + columnFunc.name + " error: " + columnFunc.error);

	//looping through kernel
	for(var i = 0; i < numIterations; i++){
		convolutionRowsGPU(rowFunc, d_Buffer, d_Input, d_Kernel, imageW, imageH);
		webcuda.synchronizeCtx();
		convolutionColumnsGPU(columnFunc, d_Output, d_Buffer, d_Kernel, imageW, imageH);
		webcuda.synchronizeCtx();
	}
	
	//Retrieving Data from CUDA Device Memory
	print("copying CUDA Mem Result to device");
	//TODO should allow them to be asynchronous
	var copyDtoH = webcuda.copyDtoH(h_Output.buffer, d_Output);
	print("copying result: " + copyDtoH);

	//Freeing CUDA Memory
	print("freeing CUDA memory");
	var memFree = webcuda.free(d_Input);
	print("d_Input free memory result: "+memFree);
	var memFree = webcuda.free(d_Output);
	print("d_Output free memory result: "+memFree);
	var memFree = webcuda.free(d_Buffer);
	print("d_Buffer free memory result: "+memFree);

	//Freeing CUDA Module
	print("freeing CUDA module");
	var moduleFree = webcuda.moduleUnload(module);
	print("free module result: " + moduleFree);

	//Destroying CUDA context
	print("destroying CUDA context");
	var ctxFree = webcuda.destroyCtx(ctx);
	print("free context result: "+ ctxFree);

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
	print("launch Result: " + launchResult);
}

function convolutionColumnsGPU(
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
	print("launch Result: " + launchResult);
}
