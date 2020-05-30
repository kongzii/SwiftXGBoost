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

RUN python3.8 -m pip install xgboost==1.1.0

COPY install.sh install.sh
RUN chmod +x install.sh && ./install.sh

WORKDIR /app

FROM build AS app

COPY . ./

CMD ["swift", "test"]
