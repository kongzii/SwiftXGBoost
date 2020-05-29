FROM swift:5.2 AS build

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        wget curl clang libblocksruntime-dev libxml2-dev \
        libxml2 zlib1g-dev git python3.8 python3.8-dev libpython3.8-dev python3-pip \
        pkg-config libatomic1 netcat-openbsd libfreetype6-dev

RUN wget https://github.com/Kitware/CMake/releases/download/v3.15.2/cmake-3.15.2.tar.gz \
    && apt-get remove --purge -y cmake \
    && tar -zxvf cmake-3.15.2.tar.gz \
    && rm cmake-3.15.2.tar.gz \
    && cd cmake-3.15.2 \
    && ./bootstrap \
    && make \
    && make install \
    && cmake --version

RUN git clone --recursive https://github.com/dmlc/xgboost \
    && cd xgboost \
    && git checkout release_1.1.0 \
    && mkdir build \
    && cd build \
    && cmake -DCMAKE_INSTALL_LIBDIR=/usr/lib .. \
    && make -j$(nproc) \
    && make install

RUN python3.8 -m pip install xgboost==1.1.0

RUN git clone https://github.com/SwiftDocOrg/swift-doc \
    && cd swift-doc \
    && make install

WORKDIR /app

FROM build AS app

COPY . ./

CMD ["swift", "test"]