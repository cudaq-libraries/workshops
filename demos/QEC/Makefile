exec = run_stabilizer_code.x

cobjects = run_stabilizer_code.o SurfaceCodeQubit.o

nvqflags = --enable-mlir --target=nvidia

$(exec): $(cobjects)
		nvq++ $(nvqflags) $(cobjects) -o $(exec)

%.o: %.cpp
		nvq++ $(nvqflags) -c -o $@ $<

clean:
		rm *.o *.x *.tmp

