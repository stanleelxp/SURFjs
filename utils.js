
    isNumber(obj) {
        return obj === +obj;
    }

    genArray(height, val) {
        if (height === undefined || height < 1 || !isNumber(val)) {
            return [];
        } else {
            let arr1D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(val);
            }
            return arr1D;
        }
    }

    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1 || !isNumber(val)) {
            return [];
        } else if (width === undefined) {
            let arr1D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(val);
            }
            return arr1D;
        } else {
            let arr1D = [];
            let arr2D = [];
            for (let i = 0; i < width; i++) {
                arr1D.push(val);
            }
            for (let i = 0; i < height; i++) {
                arr2D.push(arr1D.slice());
            }
            return arr2D;
        }
    }

    genEye(n) {
        if (n === undefined || N < 1) {
            return [];
        } else {
            let mat = this.genMatrix(n, n, 0);
            for (let i = 0; i < n; i++) {
                mat[i][i] = 1;
            }
            return mat;
        }
    }

    mat2Array(mat) {
        const height = src.length;
        const width = src[0].length;
        let arr = this.genArray(height * width, 0);
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                arr[i * width + j] = mat[i][j];
            }
        }
        return arr;
    }

    genZeros(height, width) {
        if (height === undefined || height < 1 || width < 1) {
            return [];
        } else if (width === undefined) {
            let arr1D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(0);
            }
            return arr1D;
        } else {
            let arr1D = [];
            let arr2D = [];
            for (let i = 0; i < width; i++) {
                arr1D.push(0);
            }
            for (let i = 0; i < height; i++) {
                arr2D.push(arr1D.slice());
            }
            return arr2D;
        }
    }

    getMax2D(src) {
        const height = src.length;
        const width = src[0].length;
        let max = src[0][0];
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                if (max < src[i][j]) {
                    max = src[i][j];
                }
            }
        }
        return max;
    }

    createGaussianKernel1D(n, SIGMA) {
        let sum = 0;
        let kernel = [];
        const sigma = Math.max(SIGMA, 0);
        const scale = 0.5 / (sigma * sigma);
        const n2 = Math.floor(0.5 * n);
        for (let i = 0; i < n; i++) {
            let d2 = (i - n2) * (i - n2);
            kernel.push(Math.exp(-d2 * scale));
            sum += kernel[i];
        }
        for (let i = 0; i < n; i++) {
            kernel[i] /= sum;
        }
        return kernel;
    }

    integral(img) {
        const height = img.bitmap.height;
        const width = img.bitmap.width;
        const data = img.bitmap.data;
        const sum = this.genZeros(height, width);
        // TODO
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                if (i === 0 && j === 0) {
                    sum[i][j] = data[0];
                } else if ( i === 0) {
                    const idx = j * 4;
                    sum[i][j] = sum[i][j - 1] + data[idx];
                } else if (j === 0) {
                    const idx = i * width * 4;
                    sum[i][j] = sum[i - 1][j] + data[idx];
                } else {
                    const idx = i * width * 4 + j * 4;
                    // FIXME can be faster
                    sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + data[idx];
                }
            }
        }
        return sum;
    }


    interpolate(k, kMin, vMin, kMax, vMax) {
        if (kMin === kMax) {
            return vMin;
        }
        return Math.round((k - kMin) * vMax + (kMax - k) * vMin);
    }

    resize(src, hDst, wDst) {
        const hSrc = src.length;
        const wSrc = src[0].length;
        let dst = this.genZeros(hDst, wDst);

        for (let i = 0; i < hDst; i++) {
            for (let j = 0; j < wDst; j++) {
                const x = (j * wSrc) / wDst;
                const xMin = Math.floor(x);
                const xMax = Math.min(Math.ceil(x), wSrc - 1);

                const y = (i * hSrc) / hDst;
                const yMin = Math.floor(y);
                const yMax = Math.min(Math.ceil(y), hSrc - 1);

                // let posMin = yMin * wSrc + xMin;
                // let posMax = yMin * wSrc + xMax;
                const vMin = this.interpolate(x, xMin, src[yMin][xMin], xMax, src[yMin][xMax]);

                // special case, y is integer
                if (yMax === yMin) {
                    dst[i][j] = vMin;
                } else {
                    // posMin = yMax * wSrc + xMin;
                    // posMax = yMax * wSrc + xMax;
                    const vMax = this.interpolate(x, xMin, src[yMax][xMin], xMax, src[yMax][xMax]);
                    dst[i][j] = this.interpolate(y, yMin, vMin, yMax, vMax);
                }
            }
        }
        return dst;
    }


    clearArray(arr) {
        arr.splice(0, arr.length);
    }
