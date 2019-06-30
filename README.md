# facadeSeg

# creat build folder
mkdir build
cd build
# create Visual Studio project files using cmake
cmake -G "Visual Studio 15 2017 Win64" -DCMAKE_PREFIX_PATH=/your/path/QT ..
# build our application
cmake --build . --config Release
