
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
	
	//Loading Module
	print("loading CUDA module");
	var module = webcuda.moduleLoad("tests/base_test/WebCUDA/base.ptx");
	print("fname: " + module.fname + " error: " + module.error);

	//Retrieving Function from Module
	print("retrieving function from module");
	var cuFunc = webcuda.getFunction(module, "base");
	print("name: " + cuFunc.name + " error: " + cuFunc.error);

	//Launching the Kernel
	print("trying to launch kernel");
	var launchResult = webcuda.launchKernel(cuFunc, [1,1,1], [1,1,1], []);
	print("launch result: " + launchResult);

	//unloading module
	var moduleFree = webcuda.moduleUnload(module);
	print("free module result: " + moduleFree);

	//erasing context
	var ctxFree = webcuda.destroyCtx(ctx);
	print("free context result: " + ctxFree);

}

main();
