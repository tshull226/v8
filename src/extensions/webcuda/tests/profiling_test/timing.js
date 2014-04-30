
print("timing started");
%BeginTiming();
var sum = 0;
for(var i = 0; i < 100000; i++){
	sum += i;
}
print("stopped timer");
var value = %GetTiming();

print("timer value (in microseconds)" + value);

