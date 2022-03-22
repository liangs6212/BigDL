#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from abc import ABCMeta, abstractmethod


class Forecaster(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, **kwargs):
        pass

    @property
    @abstractmethod
    def evaluate(self, **kwargs):
        pass

    @evaluate.getter
    @abstractmethod
    def evaluate(self, **kwargs):
        if not self.distributed and not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling predict!")

    @property
    @abstractmethod
    def predict(self, **kwargs):
        pass

    @predict.getter
    @abstractmethod
    def predict(self, **kwargs):
        if not self.distributed and not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling predict!")

    @property
    @abstractmethod
    def save(self, **kwargs):
        pass

    @save.getter
    @abstractmethod
    def save(self, **kwargs):
        if not self.distributed and not self.fitted:
            raise RuntimeError("You must call fit or restore first before calling predict!")

    @property
    @abstractmethod
    def quantize(self, **kwargs):
        pass

    @quantize.getter
    @abstractmethod
    def quantize(self, **kwargs):
        # check model support for quantization
        if not self.quantize_available:
            raise NotImplementedError("This model has not supported quantization.")

        # Distributed forecaster does not support quantization
        if self.distributed:
            raise NotImplementedError("quantization has not been supported for distributed "
                                      "forecaster. You can call .to_local() to transform the "
                                      "forecaster to a non-distributed version.")
