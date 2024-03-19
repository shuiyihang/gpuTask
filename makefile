CC = nvcc
SOURCE_FILE = edge_detect.cu

TARGETBIN := ./detect_work

CXXFLAGS = -w

INCLUDES +=-I  /usr/include/opencv4/ -I /usr/local/include #编译头文件目录
LIBS += -L/usr/lib/aarch64-linux-gnu -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_objdetect #链接具体使用的库

$(TARGETBIN):$(SOURCE_FILE)
	@$(CC)  $(SOURCE_FILE) -o $@ ${INCLUDES}  ${LIBS} ${CXXFLAGS}

.PHONY:clean
clean:
	-rm -rf $(TARGETBIN)



