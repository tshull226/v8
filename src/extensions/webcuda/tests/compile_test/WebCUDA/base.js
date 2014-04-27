
function main(){
	runCuda();

	print("PASSED");
}

function runCuda(seed){
	//Retrieving Device
	print("retrieving Device Info");
	var dev = webcuda.Device(0);

	//Setting up Context for CUDA
	print("creating Context");
	var ctx = webcuda.Context(0, dev);

	/*******************TEST 1*********************/
	print("test 1");
	//Loading Module
	print("loading CUDA module");
	var module = webcuda.compileFile("tests/compile_test/WebCUDA/base");
	print("fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "base");
	print("name: " + cuFunc.name + " error: " + cuFunc.error);

	//Launching the Kernel
	print("trying to launch kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [1,1,1], [1,1,1], []);
	print("launch result: " + launchResult);

	//freeing module
	var moduleFree = webcuda.moduleUnload(module);
	print("free module result: " + moduleFree);
	
	/*******************TEST 2*********************/
	print("test 2");
	//Loading Module
	print("loading CUDA module");
	module = webcuda.compileText("tests/compile_test/WebCUDA/base");
	print("fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	cuFunc = webcuda.getFunction(module, "base");
	print("name: " + cuFunc.name + " error: " + cuFunc.error);

	//Launching the Kernel
	print("trying to launch kernel");
	launchResult = webcuda.launchKernel(cuFunc, [1,1,1], [1,1,1], []);
	print("launch result: " + launchResult);

	//freeing module
	moduleFree = webcuda.moduleUnload(module);
	print("free module result: " + moduleFree);

	//erasing context
	var ctxFree = webcuda.destroyCtx(ctx);
	print("free context result: " + ctxFree);

}

main();
