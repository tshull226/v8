var imageW = 8192;
var imageH = 8192;
var numIterations = 1;
var KERNEL_RADIUS = 8
var KERNEL_LENGTH = 2 * KERNEL_RADIUS + 1

function generateRandom(size, profiler){
	profiler.start("Allocating host memory");
	var output = new Float32Array(size);
	profiler.stop("Allocating host memory");
	var i;
	for(i = 0; i < size; i++){
		output[i] = (Math.random() % 16);
	}
	return output;
}

function main(){
	load("tests/Profiler/Profiler.js")
	var profiler = new Profiler();
	var h_Kernel = generateRandom(KERNEL_LENGTH, profiler);
	var h_Input = generateRandom(imageW * imageH, profiler);
	//profiler.start("Total");
	runJS(h_Kernel, h_Input, profiler);
	//profiler.stop("Total");
	profiler.print();
}

function runJS(h_Kernel, h_Input, profiler){
	print("Starting JavaScript Execution!");
	profiler.start("Allocating host memory");
	var h_Buffer = new Float32Array(imageW * imageH);
	var h_Output = new Float32Array(imageW * imageH);
	profiler.stop("Allocating host memory");
	
	profiler.start("Computation");
	for(var i = 0; i < numIterations; i++){
		convolutionRowJS(h_Buffer, h_Input, h_Kernel, imageW, imageH, KERNEL_RADIUS);
		convolutionColumnJS(h_Output, h_Buffer, h_Kernel, imageW, imageH, KERNEL_RADIUS);
	}
	profiler.stop("Computation");

	/*
	for(i = 0; i < imageW * imageH; i++){
		print(h_Output[i]);

	}
	*/

	print("done");
}


////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
function convolutionRowJS(
		h_Dst,
		h_Src,
		h_Kernel,
		imageW,
		imageH,
		kernelR
		)
{
	var y, x, sum, k, d;
	for (y = 0; y < imageH; y++)
		for (x = 0; x < imageW; x++)
		{
			var sum = 0;

			for (k = -kernelR; k <= kernelR; k++)
			{
				d = x + k;

				if (d >= 0 && d < imageW)
					sum += h_Src[y * imageW + d] * h_Kernel[kernelR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}



////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
function convolutionColumnJS(
		h_Dst,
		h_Src,
		h_Kernel,
		imageW,
		imageH,
		kernelR
		)
{
	var y, x, sum, k, d;
	for (y = 0; y < imageH; y++)
		for (x = 0; x < imageW; x++)
		{
			sum = 0;

			for (k = -kernelR; k <= kernelR; k++)
			{
				d = y + k;

				if (d >= 0 && d < imageH)
					sum += h_Src[d * imageW + x] * h_Kernel[kernelR - k];
			}

			h_Dst[y * imageW + x] = sum;
		}
}

main();
