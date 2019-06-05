// import Jimp from 'jimp';
// import { solve } from 'ml-matrix'; // ~6.0.0
const Jimp = require('jimp');
// const matrix = require('ml-matrix'); // ~6.0.0
const { Matrix, solve, inverse } = require('ml-matrix'); // ~6.0.0

const ORI_RADIUS = 6;
const ORI_WIN = 60;
const PATCH_SZ = 20;
const SURF_ORI_SEARCH_INC = 5;
const SURF_ORI_SIGMA = 2.5;
const SURF_DESC_SIGMA = 3.3;
const SURF_HAAR_SIZE0 = 9;
const SURF_HAAR_SIZE_INC = 6;

const OK = 0;
const ERR_NEED_MORE_IMGS = 1;
const ERR_HOMOGRAPHY_EST_FAIL = 2;
const ERR_CAMERA_PARAMS_ADJUST_FAIL = 3;

const PANORAMA = 0;
const SCANS = 1;

let imgCount = 0;

class SURF {
    constructor() {
        this.sum = [];
        this.maskSum = [];
        this.mask = [];
        this.sizes = [];
        this.dets = [];
        this.traces = [];
        this.descriptors = [];
        this.middleIndices = [];
        this.keypoints = [];
        this.feature = {};
        this.octaves = 4;
        this.octaveLayers = 2;
        this.sampleStep = 2;
        this.hessianThreshold = 100;
        this.useProvidedKeypoints = false;
        // upright = true is much faster (not compute orientation)
        this.upright = true;
        this.extended = false;
        this.doDescriptors = true;
        this.filePath = '';
        this.height = 0;
        this.width = 0;
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

    calcHaarPattern(src, hf, N, offset) {
        let d = 0;
        const width = src[0].length;
        // const width = this.width;
        // console.log('width ', N, hf, offset);
        // console.log('offset ', offset);
        // console.log('hf', hf);
        const { hOffset, wOffset } = offset;
        if (width) {
            for (let k = 0; k < N; k++) {
                // d += (src[hf[k][0]] + src[hf[k][3]] - src[hf[k][1]] - src[hf[k][2]]) * hf[k][4];
                // TODO wOffset
                d += (src[hOffset + Math.floor(hf[k][0] / width)][(wOffset + hf[k][0]) % width] + 
                    src[hOffset + Math.floor(hf[k][3] / width)][(wOffset + hf[k][3]) % width] - 
                    src[hOffset + Math.floor(hf[k][1] / width)][(wOffset + hf[k][1]) % width] - 
                    src[hOffset + Math.floor(hf[k][2] / width)][(wOffset + hf[k][2]) % width]) * hf[k][4]; 
                // console.log('d0 ', src[hOffset + Math.floor(hf[k][0] / width)][(wOffset + hf[k][0]) % width]);
                // console.log('d1 ', src[hOffset + Math.floor(hf[k][1] / width)][(wOffset + hf[k][1]) % width]);
                // console.log('d2 ', src[hOffset + Math.floor(hf[k][2] / width)][(wOffset + hf[k][2]) % width]);
                // console.log('d3 ', src[hOffset + Math.floor(hf[k][3] / width)][(wOffset + hf[k][3]) % width]);
            }
        }
        // console.log('d ', d);
        return d;
    }

    // dst === idx
    resizeHaarPattern(src, dst, oldSize, newSize, width) {
        const ratio = newSize / oldSize;
        for (let k = 0; k < dst.length; k++) {
            const dx1 = Math.round(ratio * src[k][0]);
            const dy1 = Math.round(ratio * src[k][1]);
            const dx2 = Math.round(ratio * src[k][2]);
            const dy2 = Math.round(ratio * src[k][3]);
            dst[k][0] = dy1 * width + dx1;
            dst[k][1] = dy2 * width + dx1;
            dst[k][2] = dy1 * width + dx2;
            dst[k][3] = dy2 * width + dx2;
            // weight
            dst[k][4] = src[k][4] / ((dx2 - dx1) * (dy2 - dy1));
        }
        // console.log('dst ', dst);
    }

    calcLayerDetAndTrace(size, sampleStep, det, trace) {
        const height = this.height;
        const width = this.width;
        // if (size > height - 1 || size > width - 1) {
        if (size > height || size > width) {
            return;
        }

        const NX = 3;
        const NY = 3;
        const NXY = 4;
        const dx_s = [[0, 2, 3, 7, 1], [3, 2, 6, 7, -2], [6, 2, 9, 7, 1]];
        const dy_s = [[2, 0, 7, 3, 1], [2, 3, 7, 6, -2], [2, 6, 7, 9, 1]];
        const dxy_s = [[1, 1, 4, 4, 1], [5, 1, 8, 4, -1], [1, 5, 4, 8, -1], [5, 5, 8, 8, 1]];

        const Dx = this.genZeros(NX, 5); 
        const Dy = this.genZeros(NY, 5); 
         const Dxy = this.genZeros(NXY, 5); 

        this.resizeHaarPattern(dx_s, Dx, 9, size, width);
        this.resizeHaarPattern(dy_s, Dy, 9, size, width);
        this.resizeHaarPattern(dxy_s, Dxy, 9, size, width);

        // const heightRange = 1 + (height - 1 - size) / sampleStep;
        // const widthRange = 1 + (width - 1 - size) / sampleStep;
        const heightRange = (height - 1 - size) / sampleStep;
        const widthRange = (width - 1 - size) / sampleStep;
        const margin = Math.round(0.5 * size / sampleStep);
        // console.log('sum  ', this.sum);
        // console.log('range ', heightRange, widthRange);
        for (let i = 0; i < heightRange; i++) {
            // let sumIdx = i * sampleStep;
            let sumIdx = i * sampleStep * width;
            let sumOffset = {
                hOffset: i * sampleStep,
                wOffset: 0 
            };
            for (let j = 0; j < widthRange; j++) {
                const dx = this.calcHaarPattern(this.sum, Dx, 3, sumOffset);
                const dy = this.calcHaarPattern(this.sum, Dy, 3, sumOffset);
                const dxy = this.calcHaarPattern(this.sum, Dxy, 4, sumOffset);
                sumIdx += sampleStep;
                sumOffset = {
                    hOffset: Math.floor(sumIdx / width),
                    wOffset: sumIdx % width
                };
                det[i + margin][j + margin] = dx * dy - 0.81 * dxy * dxy;
                // let aa = dx * dy - 0.81 * dxy * dxy;
                // console.log('aa ', dx, dy, dxy);
                trace[i + margin][j + margin] = dx + dy;
            }
        }
    }

    interpolateKeypoint(N9, dx, dy, ds, kpt) {
        // const b = [-(N9[1][5] - N9[1][3]) * 0.5, -(N9[1][7] - N9[1][1]) * 0.5, -(N9[2][4] - N9[0][4]) * 0.5];
        const b = [[-(N9[1][5] - N9[1][3]) * 0.5], [-(N9[1][7] - N9[1][1]) * 0.5], [-(N9[2][4] - N9[0][4]) * 0.5]];
        const A = [[N9[1][3] - 2 * N9[1][4] + N9[1][5],
            (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25,
            (N9[2][5] - N9[2][3] - N9[0][5] + N9[0][3]) * 0.25],
            [(N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25,
            N9[1][1] - 2 * N9[1][4] + N9[1][7],
            (N9[2][7] - N9[2][1] - N9[0][7] + N9[0][1]) * 0.25],
            [(N9[2][5] - N9[2][3] - N9[0][5] + N9[0][3]) * 0.25,
            (N9[2][7] - N9[2][1] - N9[0][7] + N9[0][1]) * 0.25,
            N9[0][4] - 2 * N9[1][4] + N9[1][4]]];

        // TODO
        // const x = A.solve(b, decomp_lu);
        // const x = matrix.solve(A, b, (useSVD = true));
        // console.log('b ', b);
        // console.log('A ', A);
        // const A_ = new Matrix(A);
        // const invA = A.pseudoInverse();
        // const x = invA.mmul(b);
        const invObject = solve(A, b);
        const x = invObject.data;
        console.log('x ', x);
        const isOK = (x[0] !== 0 || x[1] !== 0 || x[2] !== 0) &&
            Math.abs(x[0]) <= 1 && Math.abs(x[1]) <= 1 && Math.abs(x[2]) <= 1;

        if (isOK) {
            kpt.x += x[0] * dx;
            kpt.y += x[1] * dy;
            kpt.size = Math.round(kpt.size + x[2] * ds);
        }
        return isOK;
    }

    buildInvoker(totalLayers) {
        for (let i = 0; i < totalLayers; i++) {
            this.calcLayerDetAndTrace(this.sizes[i], this.sampleSteps[i], this.dets[i], this.traces[i]);
        }

    }

    // findInvoker(sum, mask_sum, dets, traces, sizes, sampleSteps, middleIndices, keypoints, octaveLayers, hessianThreshold) {
    findInvoker(middleLayers) {
        for (let i = 0; i < middleLayers; i++) {
            const octave = i / this.octaveLayers;
            const layer = this.middleIndices[i];
            this.findMaxInLayer(octave, layer);
        }

    }

    // findMaxInLayer(sum, mask_sum, dets, traces, sizes, keypoints, octave, layer, hessianThreshold, sampleStep) {
    findMaxInLayer(octave, layer) {
        const dm = [0, 0, 9, 9, 1];
        const Dm = this.genZeros(1, 5);
        const dets = this.dets;
        const sizes = this.sizes;
        const traces = this.traces;
        const maskSum = this.maskSum;
        const size = this.sizes[layer];
        const sampleStep = this.sampleSteps[layer];
        const height = this.height;
        const width = this.width;
        const layerHeight = height / sampleStep;
        const layerWidth = width / sampleStep;
        const hessianThreshold = this.hessianThreshold;
        // const margin = (sizes[layer + 1] / 2) / sampleStep + 1;
        const margin = Math.floor((sizes[layer + 1] * 0.5) / sampleStep + 1);
        // console.log('oct ', size);
        // console.log('dm ', maskSum);
        if (!maskSum) {
            this.resizeHaarPattern(dm, Dm, 9, size, maskSum[0].length);
        }

        // TODO
        // const step = Math.floor(dets[layer].step / dets[layer].elemSize());
        // const step = width;
        // const step = dets[layer][0].length;
        // console.log('step ', step);
        // console.log('layer ', layer);
        // console.log('margin', margin);

        // for (let i = 0; i < layerHeight; i++) {
        //     for (let j = 0; j < layerWidth; j++) {
        for (let i = margin; i < layerHeight - margin; i++) {
             for (let j = margin; j < layerWidth - margin; j++) {
                 const val = dets[layer][i][j];
                 // console.log('val ', val);
                 if (val > hessianThreshold) {
                     // const hOffset = sampleStep * (i - size * 0.5 / sampleStep);
                     // const wOffset = sampleStep * (j - size * 0.5 / sampleStep);
                     const hOffset = sampleStep * (i - size * 0.5 / sampleStep);
                     const wOffset = sampleStep * (j - size * 0.5 / sampleStep);
                     const d1 = dets[layer - 1];
                     const d2 = dets[layer];
                     const d3 = dets[layer + 1];

                     /*
                    const N9 = [[dets[layer - 1][-step - 1], dets[layer - 1][-step], dets[layer - 1][-step + 1],
                        dets[layer - 1][-1], dets[layer - 1][0], dets[layer - 1][1],
                        dets[layer - 1][step - 1], dets[layer - 1][step], dets[layer - 1][step + 1]],
                        [dets[layer1][-step - 1], dets[layer1][-step], dets[layer][-step + 1],
                            dets[layer][-1], dets[layer][0], dets[layer][1],
                            dets[layer][step - 1], dets[layer][step], dets[layer][step + 1]],
                        [dets[layer + 1][-step - 1], dets[layer + 1][-step], dets[layer + 1][-step + 1],
                            dets[layer + 1][-1], dets[layer + 1][0], dets[layer + 1][1],
                            dets[layer + 1][step - 1], dets[layer + 1][step], dets[layer + 1][step + 1]]];
                            */
                     // TODO -step
                     /*
                     const N9 = [[d1[-step - 1], d1[-step], d1[-step + 1],
                         d1[-1], d1[0], d1[1],
                         d1[step - 1], d1[step], d1[step + 1]],
                         [d2[-step - 1], d2[-step], d2[-step + 1],
                             d2[-1], d2[0], d2[1],
                             d2[step - 1], d2[step], d2[step + 1]],
                         [d3[-step - 1], d3[-step], d3[-step + 1],
                             d3[-1], d3[0], d3[1],
                             d3[step - 1], d3[step], d3[step + 1]]];
                             */

                     const N9 = [[d1[i - 1][j - 1], d1[i - 1][j], d1[i - 1][j + 1],
                         d1[i][j - 1], d2[i][j], d2[i][j + 1],
                         d1[i + 1][j - 1], d2[i + 1][j], d2[i + 1][j + 1]],
                         [d2[i - 1][j - 1], d2[i - 1][j], d2[i - 1][j + 1],
                         d2[i][j - 1], d2[i][j], d2[i][j + 1],
                         d2[i + 1][j - 1], d2[i + 1][j], d2[i + 1][j + 1]],
                         [d3[i - 1][j - 1], d3[i - 1][j], d3[i - 1][j + 1],
                         d3[i][j - 1], d3[i][j], d3[i][j + 1],
                         d3[i + 1][j - 1], d3[i + 1][j], d3[i + 1][j + 1]]];

                     // console.log('N9 ', N9);
                     // if (!maskSum[0]) {
                     if (!maskSum) {
                         const maskSumOffset = {
                             hOffset: hOffset,
                             wOffset: wOffset 
                         };
                         console.log('maskSumOffset ', maskSumOffset);
                         const mval = this.calcHaarPattern(maskSum, Dm, 1, maskSumOffset);
                         console.log('mval ', mval);
                         if (mval < 0.5) {
                             continue;
                         }
                     }

                     // console.log('val ', val);
                     const maxInN9 = this.getMax2D(N9);
                     // console.log('mval ', maxInN9);
                     // TODO N9 max
                     if(val > (1 - 1e-6) * maxInN9) {
                         // console.log('val ', val);
                         // console.log('mval ', maxInN9);
                         const hCenter = hOffset + (size - 1) * 0.5;
                         const wCenter = wOffset + (size - 1) * 0.5;
                         // console.log('hoffset', hOffset);

                         // const kpt = this.genKeyPoint(hCenter, wCenter, sizes[layer], -1, val, octave, (traces[layer][j] > 0) - (traces[layer][j] < 0));
                         const kpt = {
                             pt: {
                                 x: wCenter,
                                 y: hCenter
                             }
                             size: sizes[layer],
                             angle: -1,
                             response: val,
                             octave: octave,
                             classID: (traces[layer][j] > 0) - (traces[layer][j] < 0)
                         };
                         const ds = size - sizes[layer - 1];
                         const interpOK = this.interpolateKeypoint(N9, sampleStep, sampleStep, ds, kpt);
                         // console.log('OK ', interpOK, size, sizes[layer - 1], ds);

                         if (interpOK) {
                             // TODO
                             // cv::AutoLock
                             this.keypoints.push(kpt);
                         }
                     }
                 }
             }
        }
    }

    clear(arr) {
        if (arr.length) {
            arr.splice(0, arr.length);
        }
    }

    // fastHessianDetector(sum, mask_sum, keypoints, octaves, octaveLayers, hessianThreshold) {
    fastHessianDetector() {
        const sampleStep0 = 1;
        const octaveLayers = this.octaveLayers;
        const octaves = this.octaves;
        const height = this.height;
        const width = this.width;
        const totalLayers = (octaveLayers + 2) * octaves;
        const middleLayers = octaveLayers * octaves;
        // this.dets = this.genMatrices(totalLayers, height, width);
        // this.traces = this.genMatrices(totalLayers, height, width);

        this.sizes = this.genZeros(totalLayers);
        this.sampleSteps = this.genZeros(totalLayers);
        this.middleIndices = this.genZeros(middleLayers);

        // clear()
        // this.keypoints.splice(0, this.keypoints.length);
        this.clear(this.keypoints);
        let index = 0;
        let middleIndex = 0;
        let step = sampleStep0;
        for (let octave = 0; octave < octaves; octave++) {
            for (let layer = 0; layer < octaveLayers + 2; layer++) {
                // slice() not needed
                this.dets.push(this.genZeros(height / step, width / step));
                this.traces.push(this.genZeros(height / step, width / step));
                this.sizes[index] = (SURF_HAAR_SIZE0 + SURF_HAAR_SIZE_INC * layer) << octave;
                this.sampleSteps[index] = step;
                if (0 < layer && layer <= octaveLayers) {
                    this.middleIndices[middleIndex] = index;
                    middleIndex += 1;
                }
                index += 1;
            }
            step *= 2;
        }

        // buildInvoker(sum, sizes, sampleSteps, dets, traces);
        // findInvoker(sum, mask_sum, dets, traces, sizes, sampleSteps, middleIndices, keypoints, octaveLayers, hessianThreshold);
        this.buildInvoker(totalLayers);
        this.findInvoker(middleLayers);
        // TODO
        // keypoints.sort();
        // FIXME
        // keypoints.sort((a, b) => a.response - b.response);
        this.keypoints.sort((a, b) => b.response - a.response);
        console.log('fH ok', this.keypoints.length);
    }
    
    invoke(img) {
        const NX = 2;
        const NY = 2;
        const dx_s =[[0, 0, 2, 4, -1], [2, 0, 4, 4, 1]];
        const dy_s =[[0, 0, 4, 2, 1], [0, 2, 4, 4, -1]];
        const oriSampleBound = (2 * ORI_RADIUS + 1) * (2 * ORI_RADIUS + 1);
        const X = this.genZeros(oriSampleBound);
        const Y = this.genZeros(oriSampleBound);
        const angle = this.genZeros(oriSampleBound);
        const DX = this.genZeros(PATCH_SZ, PATCH_SZ);
        const DY = this.genZeros(PATCH_SZ, PATCH_SZ);
        // const patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        // const _patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        const apt = this.genZeros(oriSampleBound);
        const aptw = this.genZeros(oriSampleBound);
        const DW = this.genZeros(PATCH_SZ * PATCH_SZ);
        const gOri = this.createGaussianKernel1D(2 * ORI_RADIUS + 1, SURF_ORI_SIGMA);
        const gDesc = this.createGaussianKernel1D(PATCH_SZ, SURF_DESC_SIGMA);
        const dSize = this.extended ? 128: 64;
        const height = this.height;
        const width = this.width;
        const kps = this.keypoints.length;
        const width1 = width - 1;
        const height1 = height - 1;
        const data = img.bitmap.data;
        let oriSamples = 0;
        /*
        let maxSize = 0;
        let iMaxSize = 0;
        for (let k = 0; k < kps; k++) {
            maxSize = Math.max(maxSize, this.keypoints[k].size);
        }
        iMaxSize = Math.max(1, Math.ceil((PATCH_SZ + 1) * maxSize * 1.2 / 9.0));
        const winBuf = this.genZeros(imaxSize, imaxSize);
        */

        for (let i = -ORI_RADIUS; i <= ORI_RADIUS; i++) {
            for (let j = -ORI_RADIUS; j <= ORI_RADIUS; j++) {
                if (i * i + j * j <= ORI_RADIUS * ORI_RADIUS) {
                    const weight_ = gOri[i + ORI_RADIUS] * gOri[j + ORI_RADIUS];
                    // TODO apt
                    // apt.push([i, j]);
                    apt[oriSamples] = [i, j];
                    aptw[oriSamples] = weight_;
                    oriSamples += 1;
                }
            }
        }
        for (let i = 0; i < PATCH_SZ; i++) {
            for (let j = 0; j < PATCH_SZ; j++) {
                DW[i * PATCH_SZ + j] = gDesc[i] * gDesc[j];
            }
        }

        for (let k = 0; k < kps; k++) {
            let descriptorDir = 360.0 - 90.0;
            let squareMag = 0;
            let dOffset = 0;
            const dx_t = this.genZeros(NX, 5);
            const dy_t = this.genZeros(NY, 5);
            const size = this.keypoints[k].size;
            const s = size * 1.2 / 9.0;
            const gradWaveSize = 2 * Math.round(2 * s);
            const center = this.keypoints[k].pt;

            if (height < gradWaveSize || width < gradWaveSize) {
                this.keypoints[k].size = -1;
                continue;
            }
            if (!this.upright) {
                let nangle = 0;
                this.resizeHaarPattern(dx_s, dx_t, 4, gradWaveSize, width); 
                this.resizeHaarPattern(dy_s, dy_t, 4, gradWaveSize, width); 
                for (let k1 = 0; k1 < oriSamples; k1++) {
                    const y = Math.round(center.y + apt[k1][0] * s - (gradWaveSize - 1) * 0.5);
                    const x = Math.round(center.x + apt[k1][1] * s - (gradWaveSize - 1) * 0.5);
                    if (y < 0 || y >= height - gradWaveSize ||
                        x < 0 || x >= width - gradWaveSize) {
                        continue;
                    }
                    // console.log('xy ', x, y);
                    const sumOffset = {
                        hOffset: y,
                        wOffset: x 
                    };
                    const vx = this.calcHaarPattern(this.sum, dx_t, 2, sumOffset);
                    const vy = this.calcHaarPattern(this.sum, dy_t, 2, sumOffset);
                    X[nangle] = vx * aptw[k1];
                    Y[nangle] = vy * aptw[k1];
                    nangle += 1;
                }
                if (nangle === 0){
                    this.keypoints[k].size = -1;
                    continue;
                }
                for (let i = 0; i < X.length; i++) {
                    angle[i] = Math.atan2(Y[i], X[i]) * 180.0 / Math.PI;
                }
                let bestX = 0;
                let bestY = 0;
                let descriptorMod = 0;
                for (let i = 0; i < 360; i += SURF_ORI_SEARCH_INC) {
                    let sumX = 0;
                    let sumY = 0;
                    let tempMod = 0;
                    for (let j = 0; j < nangle; j++) {
                        let d = Math.abs(Math.round(angle[j]) - i);
                        if (d < ORI_WIN / 2 || d < 360 - ORI_WIN / 2) {
                            sumX += X[j];
                            sumY += Y[j];
                        }
                    }
                    tempMod = sumX * sumX + sumY * sumY;
                    if (tempMod > descriptorMod) {
                        descriptorMod = tempMod;
                        bestX = sumX;
                        bestY = sumY;
                    }
                }
                descriptorDir = Math.atan2(-bestY, bestX);
            }
            this.keypoints[k].angle = descriptorDir;
            if (!this.descriptors || !this.descriptors[0]) {
                continue;
            }

            const winSize = Math.floor((PATCH_SZ + 1) * s);
            // console.log('winsize ', winSize);
            let win = this.genZeros(winSize, winSize);
            if (!this.upright) {
                descriptorDir *= Math.PI / 180.0;
                const sinDir = - Math.sin(descriptorDir);
                const cosDir = Math.cos(descriptorDir);
                const winOffset = -(winSize - 1) * 0.5;
                let startX = center.x + winOffset * cosDir + winOffset * sinDir;
                let startY = center.x - winOffset * cosDir + winOffset * sinDir;
                for (let i = 0; i < winSize; i++, startX += sinDir, startY += cosDir) {
                    let pixelX = startX;
                    let pixelY = startY;
                    for (let j = 0; j < winSize; j++, pixelX += cosDir, pixelY -= sinDir) {
                        const ix = Math.floor(pixelX);
                        const iy = Math.floor(pixelY);
                        const idx = iy * width * 4 + ix;
                        if (ix < width1 && iy < height1) {
                            const a = pixelX - ix;
                            const b = pixelY - iy;
                            win[i][j] = Math.round(data[idx] * (1.0 - a) * (1.0 - b) + data[idx + 1] * a * (1.0 - b) + 
                                data[idx + width * 4] * (1.0 - a) * b + data[idx + width * 4 + 1] * a * (1.0 - b));
                        } else {
                            const x = Math.min(Math.max(Math.round(pixelX), 0), width1);
                            const y = Math.min(Math.max(Math.round(pixelY), 0), height1);
                            win[i][j] =  data[y * width * 4 + x];
                        }
                    }
                }
            } else {
                const winOffset = -(winSize - 1) * 0.5;
                let startX = Math.round(center.x + winOffset);
                let startY = Math.round(center.y - winOffset);
                for (let i = 0; i < winSize; i++, startX++) {
                    let pixelX = startX;
                    let pixelY = startY;
                    for (let j = 0; j < winSize; j++, pixelY--) {
                        let x = Math.max(pixelX, 0);
                        let y = Math.max(pixelY, 0);
                        x = Math.min(x, width1);
                        y = Math.min(y, height1);
                        // win[i][j] =  data[y][x];
                        win[i][j] =  data[y * width * 4 + x];
                    }
                }
            }
            const patch = this.resize(win, PATCH_SZ + 1, PATCH_SZ + 1);
            for (let i =  0; i < PATCH_SZ; i++) {
                for (let j =  0; j < PATCH_SZ; j++) {
                    const dw = DW[i * PATCH_SZ + j];
                    const vx = (patch[i][j + 1] - patch[i][j] + patch[i + 1][j + 1] - patch[i + 1][j]) * dw;
                    const vy = (patch[i + 1][j] - patch[i][j] + patch[i + 1][j + 1] - patch[i][j + 1]) * dw;
                    DX[i][j] = vx;
                    DY[i][j] = vy;
                }
            }
            // console.log('DX ', DX);
            // console.log('DY ', DY);
            for (let k1 = 0; k1 < dSize; k1++) {
                this.descriptors[k][k1] = 0;
            }
            if (this.extended) {
                for (let i = 0; i < 4; i++) {
                    for (let j = 0; j < 4; j++) {
                        for (let y = i * 5; y < i * 5 + 5; y++) {
                            for (let x = j * 5; x < j * 5 + 5; x++) {
                                const tx = DX[y][x];
                                const ty = DY[y][x];
                                // TODO fabs
                                if (ty >= 0) {
                                    this.descriptors[k][0 + dOffset] += tx;
                                    this.descriptors[k][1 + dOffset] += Math.abs(tx);
                                } else {
                                    this.descriptors[k][2 + dOffset] += tx;
                                    this.descriptors[k][3 + dOffset] += Math.abs(tx);
                                }
                                if (tx >= 0) {
                                    this.descriptors[k][4 + dOffset] += ty;
                                    this.descriptors[k][5 + dOffset] += Math.abs(ty);
                                } else {
                                    this.descriptors[k][6 + dOffset] += ty;
                                    this.descriptors[k][7 + dOffset] += Math.abs(ty);
                                }
                            }
                        }
                        for (let k1 = 0; k1 < 8; k1++) {
                            squareMag += this.descriptors[k][k1] * this.descriptors[k][k1];
                        }
                        dOffset += 8;
                    }
                }
            } else {
                for (let i = 0; i < 4; i++) {
                    for (let j = 0; j < 4; j++) {
                        for (let y = i * 5; y < i * 5 + 5; y++) {
                            for (let x = j * 5; x < j * 5 + 5; x++) {
                                const tx = DX[y][x];
                                const ty = DY[y][x];
                                this.descriptors[k][0 + dOffset] += tx;
                                this.descriptors[k][1 + dOffset] += ty;
                                this.descriptors[k][2 + dOffset] += Math.abs(tx);
                                this.descriptors[k][3 + dOffset] += Math.abs(ty);
                            }
                        }
                        for (let k1 = 0; k1 < 4; k1++) {
                            squareMag += [k1] * [k1];
                        }
                        dOffset += 4;
                    }
                }
            }
            const scale = 1.0 / (Math.sqrt(squareMag + 1e-6));
            for (let k1 = 0; k1 < dSize; k1++) {
                this.descriptors[k][k1] *= scale;
                // console.log('ds ', this.descriptors);
            }
        }
    }
    
    matrixScalarCompare(mat, scalar, type) {
        // const height = mat.length;
        // const width = mat[0].length;
        const height = this.height;
        const width = this.width;
        const res = this.genZeros(height, width);
        console.log('DC1 ', mat);
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < height; j++) {
                console.log('wdith', width);
                const val = mat[i][j];
                if (type === 'min') {
                    console.log('wdith', width);
                    res[i][j] = val < scalar ? val : scalar;
                } else if (type === 'max') {
                    res[i][j] = val > scalar ? val : scalar;
                } else {
                    res[i][j] = val;
                }
            }
        }
        console.log('DC1 ');
        return res;
    }

    // detectAndCompute(mask, keypoints, descriptors, useProvidedKeypoints) {
    async detectAndCompute() {
        let img = await Jimp.read(this.filePath);
        img = img.greyscale();
        // console.log('img ', img);
        this.height = img.bitmap.height;
        this.width = img.bitmap.width;
        this.sum = await this.integral(img);
        // console.log('sum ', this.sum);
        if (!this.useProvidedKeypoints) {
            // console.log('mask ', !this.mask);
            // console.log('mask0 ', !this.mask[0]);
            if (!this.mask) {
                const mask1 = this.matrixScalarCompare(this.mask, 1, 'min');
                console.log('DC2 ');
                this.maskSum = this.integral(mask1);
                console.log('maskSum ', this.maskSum);
            }
            // console.log('sum ', this.maskSum);
            this.fastHessianDetector();
            console.log('mask', !this.mask);
            if (!this.mask) {
                for (let i = 0; i < this.keypoints.length;) {
                    const { x, y } = this.keypoints[i].pt;
                    if (this.mask[y][x] === 0) {
                        // TODO
                        // erase
                        this.keypoints.splice(i, 1);
                        continue; // keep "i"
                    }
                    i++;
                }
            }
        }
        let kps = this.keypoints.length;
        console.log('kps ', kps);
        const doDescriptors = this.doDescriptors;
        if (kps > 0) {
            const dWidth = this.extended ? 128: 64;
            if (doDescriptors) {
                this.descriptors = this.genZeros(kps, dWidth);
            }
            this.invoke(img);
            let j = 0;
            for (let i = 0; i < kps; i++) {
                // console.log('i ', this.keypoints[i].size);
                if (this.keypoints[i].size === -1) {
                    console.log('i ', i);
                }
                // console.log('kpsi ', this.keypoints);
                // console.log('kpsi ', this.keypoints[i].size);
                if (this.keypoints[i].size > 0) {
                    if (i > j) {
                        console.log('j ', j);
                        this.keypoints[j] = this.keypoints[i];
                        if (doDescriptors) {
                            // TODO
                            // memcpyt(dst, src, ...);
                            // memcpy(descriptors.ptr(j), descriptors.ptr(i), dsize);
                            this.decriptors[j] = this.descriptors[i];
                            console.log('ds ', this.descriptors);
                        }
                    }
                    j += 1;
                }
            }
            if (kps > j) {
                kps = j;
                console.log('j ', j);
                console.log('kps1', kps);
                // this.keypoints.resize(kps);
                this.keypoints.splice(kps, this.keypoints.length - kps);
                if (doDescriptors) {
                    let d = [];
                    for (let k = 0; k < kps; k++) {
                        // TODO
                        // d.push(this.descriptors[k]);
                        d.push(this.descriptors[k].slice());
                    }
                    this.descriptors = d;
                    console.log('ds ', this.descriptors);
                }
            }
        }
    }

    // computeImageFeatures(finder, img, feature) {
    getFeatures(imgIdx) {
        this.feature = {
            imgIdx: imgIdx, 
            height: this.height, 
            width: this.width, 
            keypoints: this.keypoints,
            descriptors: this.descriptors
        }
    }


    async draw() {
        let img = await Jimp.read(this.filePath);
        const width = img.bitmap.width;
        const kps = this.keypoints.length;
        console.log('draw kps ', kps);
        for (let i = 0; i < kps; i++) {
            const kp = this.keypoints[i];
            const idx = Math.round(kp.y) * width * 4 + Math.round(kp.x) * 4;
            // console.log('idx ', idx);
            img.bitmap.data[idx + 0] = 255;
            img.bitmap.data[idx + 1] = 0;
            img.bitmap.data[idx + 2] = 0;
            img.bitmap.data[idx + 3] = 255;
        }
        img.write(`./test_${imgCount}.png`, () => {
            console.log('draw OK ');
        });
        imgCount += 1;
        console.log('count ', imgCount);
    }

    // create(hessianThreshold, octaves, layers, extended, upright) {
    async create(filePath, options) {
        if (!filePath) {
            return;
        }
        // default: 100, 4, 2, 64, true
        this.filePath = filePath;
        this.hessianThreshold = options.hessianThreshold || 100;
        this.octaves = options.octaves || 4;
        this.layers = options.layers || 2;
        this.extended = options.extended || false;
        this.upright = options.upright || false;
        console.log('options', this.hessianThreshold);

        await this.detectAndCompute();
        this.getFeatures(imgCount);
        await this.draw();
    }
};


class stitch {
    constructor() {
        this.features = [];
        this.hDst = [];
        this.hMask = [];
        this.cameras = [];
        this.numImages = 0;
        this.seamWorkAspect = 0;
        this.images = [];
        this.imagesWarped = [];
        this.nearPairs = [];
        this.pairMatches = [];
        this.matchesConf = 0;
        // TODO
        this.matchInfo = [];

        matchesInfo;
        FeaturesMatcher;
        AffineBestOf2NearestMatcher;
        ExposureCompensator
        GainComponsator

    }


    collectGarbage() {
    }

    matchesGraphAsString(imgNames, pairwiseMatches, confThresh) {
    }

    leaveBiggestComponent(pairwiseMatches, confThresh) {
        const numImages = this.features.length;
        //TODO
        DisjointSets comps(numImages);
        for (let i = 0; i < numImages; i++) {
            for (let j = 0; j < numImages; j++) {
                if (pairwiseMatches[i * numImages + j].confidence < confThreshold) {
                    continue;
                }
                const comp1 = comps.findSetByElem(i);
                const comp2 = comps.findSetByElem(j);
                if (comp1 != comp2) {
                    comps.mergeSets(comp1, comp2);
                }
            }
        }
        // TODO getMax
        const maxComp = getMax(comps);

        let indices = [];
        let indicesRemoved = [];
        for (let i = 0; i < numImages; i++) {
            if (comps.findSetByelem(i) == maxComp) {
                indices.push(i);
            } else {
                indicesRemoved.push(i);
            }
        }
        let subFeatures = [];
        let subPairwiseMatches = [];
        for (let i = 0; i < indices.length; i++) {
            subFeatures.push(this.features[indices[i]]);
            for (let j = 0; j < indices.length; j++) {
                subPairwiseMatches.push(pairMatches[indices[i] * numImages + indices[j]]);
                let newLength = subPairwiseMatches.length;
                subPairwiseMatches[newLength - 1].srcIdx = i;
                subPairwiseMatches[newLength - 1].dstIdx = j;
            }
        }

        if (subFeatures.length === numImages) {
            return indices;
        }

        for (let i = 0; i < indicesRemoved.length(); i++) {
            console.log('remove ', indicesRemoved[i]);
        }

        this.features = subFeatures;
        pairwiseMatches = subPairwiseMatches;
        
        return indices;
    }

    findMaxSpanningTree() {
    }

    createDefault(exposCompType) {
    }

    setNrFeeds(exposCompNrFeeds) {
    }

    setNrGainsFilteringIterations(exposCompNrFiltering) {
    }

    setBlockSize(exposCompBlockSize, exposCompBlockSize) {
    }

    setNumBands() {
    }

    setSharpness() {
    }

    feed(img, mask, dstROI, pt) {
        const height = img.length;
        const width = img[0].length;
        const dx = pt.x - dstROI.x;
        const dy = pt.y - dstROI.y;
        for (let y = 0; y < height; y++) {
            const hSrc = img[y]; 
            const hDst = dst[y + dy];
            const hMask = mask[y + dy];
            for (let x = 0; x < height; x++) {
                if (hMask[x]) {
                    hDst[x + dx] = hSrc[x];
                }
                hMask[d + dx] |= hMask[x];
            }
        }
        this.hDst = hDst;
        this.hMask = hMask;
    }

    dilate(masksWarped, dilatedMask, mat) {
    }
    
    resize(dilatedMask, seamMask, size) {
    }

    prepare() {
    }

    initialize(corners, sizes) {
    }

    process() {
    }

    warp() {
    }

    rotationWarp() {
        for (let i = 0; i < this.numImages; i++) {
            let ck = this.cameras[i];
            const swa = this.seamWorkAspect;
            ck[0][0] *= swa;
            ck[0][2] *= swa;
            ck[1][1] *= swa;
            ck[1][2] *= swa;
            
            const srcSize = {
                height: this.images[i].length,
                width: this.images[i][0].length
            };
            this.corners[i] = this.warp(srcSize, ck, this.cameras[i].R, imagesWarped[i]);

        }
    }

    compensate() {
    }


    swap(a, b) {
        let c = a;
        a = b;
        b = c;
    }

    matchPair() {
        // TODO random
        //
        const numImages = this.numImages;
        for (let i = 0; i < numImages; i++) {
            const from = this.nearPairs[i].from;
            const to = this.nearPairs[i].to;
            const pairIdx = from * numImages + to;
            this.pairMatches[pairIdx].srcIdx = from;
            this.pairMatches[pairIdx].dstIdx = to;
            const dualPairIdx = to * numImages + from; 
            // TODO FIXME
            this.pairMatches[dualPairIdx] = this.pairMatches[pairIdx];

            this.pairMatches[dualPairIdx].srcIdx = to;
            this.pairMatches[dualPairIdx].dstIdx = from;
            if (this.pairMatches[pairIdx].H) {
                // TODO inv
                this.pairMatches[dualPairIdx].H = this.pairMatches[pairIdx].H.inv();
            }

            // TODO matches
            for (let j = 0; j < this.pairMatches[dualPairIdx].matches.length; j++) {
                this.swap(this.pairMatches[dualPairIdx].matches[j].queryIdx,
                    this.pairMatches[dualPairIdx].matches[j].trainIdx);
            }
        }
    }

    match(features1, features2, matchesInfo) {
        // TODO CV_INSTRUMENT_REGION();
        // TODO clear
        // this.clear(this.matchesInfo.matches);
        this.matchesInfo.matches.clear();
        // TODO cv:DescriptorMatcher matcher
        // TODO FLANN
        let idxParams = this.KDTreeIdxParams;
        let searchParams = this.searchParams;
        
        let matches = new MatchesSet();
        let pairMatches = new DMatch();
        matcher.knnMatch(features1.descriptors, features2.descriptors, pairMatches, 2);
        for (let i = 0; i < pairMatches.length; i++) {
            if (pairMatches[i].length < 2) {
                continue;
            }
            const m0 = pairMatches[i][0];
            const m1 = pairMatches[i][1];
            if (m0.distance < (1.0 - this.matchConf) * m1.distance) {
                matchesInfo.matches.push(m0);
                // TODO insert makePair
                // matches.insert(makePair(m0.queryIdx, m0.trainIdx));
                matches.push([m0.queryIdx, m0.trainIdx]);
            }
        }

        pairMatches.clear();
        // feature2 -> 1
        matcher.knnMatch(features2.descriptors, features1.descriptors, pairMatches, 2);
        for (let i = 0; i < pairMatches.length; i++) {
            if (pairMatches[i].length < 2) {
                continue;
            }
            const m0 = pairMatches[i][0];
            const m1 = pairMatches[i][1];
            if (m0.distance < (1.0 - this.matchConf) * m1.distance) {
                // TODO
                if ([m0.queryIdx, m0.trainIdx] not in matches) {
                    matchInfo.matches.push(new DMatch(m0.trainIx, m0.queryIdx, m0.distance));
                }
            }
        }
    }




        

    findSeam() {
    }

    blend() {
    }




};



class MatchesSet {
    constructor() {
    }

    knnMatch() {
    }
}


class DMatch {
    constructor() {
    }
}


class RNG {
}

class FlannBasedMatcher {
}


class MatchesInfo {
    constructor() {
        this.srcIdx = 0;
        this.dstIdx = 0;
        this.matches = [];
        this.inliersMask = [];
        this.numInliers = [];
        this.H = [];
        this.confidence = [];
    }
}

class KeyPoint {
    constructor() {
        this.keypointExample = {
            pt: {
                x: 0,
                y: 0 
            }
            size: 0,
            angle: 0,
            response: 0,
            octave: 0,
            classID: 0 
        };
    }
}

class Pair {
}

class MatchPairsBody {
}

class Feature2D {
}

class ImageFeatures {
}

class FeaturesMatcher {
    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1) {
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

    makePair(a, b) {
    }

    operator (features, pairMatches, mask) {
        const numImages = features.length;
        const mask_ = this.genMatrix(numImages, numImages, val);
        const nearPairs = new Pair();
        for (let i = 0; i < numImages; i++) {
            for (let j = i - 1; j < numImages; j++) {
                if (features[i].keypoints.length > 0 && features[j].keypoints.length > 0 && mask_[i][j]) {
                    nearPairs.push(this.makePair(i, j));
                }
            }
        }

        pairMatches.clear();
        pairMatches.resize(numImages * numImages);
        let body = new MatchPairsBody(this, features, pairMatches, nearPairs);
        // TODO
        body(0, nearPairs.length);
    }
}

class BestOf2NearestMatcher {
    constructor() {
        this.matchConf = 0;
        this.numMatchesThresh1 = 0;
        this.numMatchesThresh2 = 0;
    }

    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1) {
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

    findHomography() {
    }
    determinant() {
    }

    match(features1, features2, matchesInfo) {
        if (matchesInfo.matches.length < this.numMatchesThresh1) {
            return;
        }

        let srcPoints = this.genMatrix(matchesInfo.matches.length, 2, 0);
        let dstPoints = this.genMatrix(matchesInfo.matches.length, 2, 0);
        for (let i = 0; i < matchesInfo.matches.length; i++) {
            const m = matchesInfo.matches[i];
            let px = features1.keypoints[m.queryIdx].x;
            let py = features1.keypoints[m.queryIdx].y;
            px -= features1.width * 0.5;
            py -= features1.height * 0.5;
            srcPoints[i][0] = px;
            srcPoints[i][1] = py;

            px = features2.keypoints[m.queryIdx].x;
            py = features2.keypoints[m.queryIdx].y;
            px -= features2.width * 0.5;
            py -= features2.height * 0.5;
            dstPoints[i][0] = px;
            dstPoints[i][1] = py;
        }
        matchesInfo.H = this.findHomography(srcPoints, dstPoints, matchesInfo.inliersMask, 'RANSAC');
        if (!matchesInfo.H || Math.abs(this.determinant(matchesInfo.H)) < 1e-6) {
            return;
        }
        matchesInfo.numInliers = 0;
        for (let i = 0; i < matchesInfo.inliersMask.length; i++) {
            if (matchesInfo,inliersMask[i]) {
                matchesInfo.numInliers += 1;
            }
        }
        matchesInfo.confidece = matchesInfo.numInliers / (8 + 0.3 * matchesInfo.matches.length);
        matchesInfo.confidece = matchesInfo.confidence > 3 ? 0.0 : matchesInfo.confidence;
        if (matchesInfo.numInliers < numMatchesThresh2) {
            return;
        }

        // for inliers
        // TODO create
        // srcPoints.create(1, matchesInfo.numInliers);
        // dstPoints.create(1, matchesInfo.numInliers);
        srcPoints = this.genMatrix(matchesInfo.numInliers, 2, 0);
        dstPoints = this.genMatrix(matchesInfo.numInliers, 2, 0);
        let inlierIdx = 0;
        for (let i = 0; i < matchesInfo.matches.length; i++) {
            if (!matchesInfo.inliersMask[i]) {
                continue;
            }
            const m = machesInfo.matches[i];
            let px = features1.keypoints[m.queryIdx].x;
            let py = features1.keypoints[m.queryIdx].y;
            px -= features1.width * 0.5;
            py -= features1.height * 0.5;
            srcPoints[i][0] = px;
            srcPoints[i][1] = py;

            px = features2.keypoints[m.queryIdx].x;
            py = features2.keypoints[m.queryIdx].y;
            px -= features2.width * 0.5;
            py -= features2.height * 0.5;
            dstPoints[i][0] = px;
            dstPoints[i][1] = py;
            inlierIdx += 1;
        }
        matchesInfo.H = this.findHomography(srcPoints, destPoints, 'RANSAC');
    }

    operator(features, pairMatches, mask, rangeWidth) {
        const numImages = features.length;
        let mask_ = this.genMatrix(numImages, numImages, 1);
        let nearPairs = new Pair();
        for (let i = 0; i < numImages - 1; i++) {
            for (let j = i + 1; j < Math.min(numImages, i + rangeWidth); j++) {
                if (features[i].keypoints.length > 0 && features[j].keypoints.length > 0 && mask_[i][j]) {
                    nearPairs.push(this.makePair(i, j));
                }
            }
        }
        pairMatches.resize(numImages * numImages);
        let body = new MatchPairsBody(this, features, pairMatches, nearPairs);
        // TODO
        body(0, nearPairs.length);
    }

    collectGarbage() {
    }
}


class Stitcher {
    constructor() {
        this.regRes = 0.6;
        this.seamEstRes = 0.1;
        this.composeRes = -1.0;
        this.confidenceThresh = 1;
        this.doWaveCorrect = false;
        this.workScale = 1;
        this.seamScale = 1;
        this.seamWorkAspect = 1;
        this.warpedImgScale = 1;
        this.imgs = [];
        this.masks = [];
        // { heidht, width }
        this.fullImgSizes = [];
        this.features = [];
        this.pairMatches = [];
        this.seamEstImgs = [];
        this.indices = [];
        this.cameras = [];
        this.resultMask = [];
        this.estimator = new what(); 
        this.bundleAdjuster = new what(); 
    }

    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1) {
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

    resize(mat) {
    }

    clear(arr) {
        arr.splice(0, arr.length);
    }

    composePanorama(images, pano) {
        const height = images[0].length;
        const width = images[0][0].length;
        // let img = this.genMatrix(height, width, 0);
        let img = [];
        this.resize(this.seamEstImgs);
        for (let i = 0; i < images.length; i++) {
            this.imgs[i] = images[i];
            img = this.resize(this.imgs[i], this.seamScale, this.seamScale);
            this.seamEstImgs[i] = img;
        }
        let subSeamEstImgs = [];
        let subImgs = [];
        for (let i = 0; i < this.indices.length; i++) {
            subImgs.push(this.imgs[indices[i]]);
            subSeamEstImgs.push(this.seamEstImgs[indices[i]]);
        }
        this.seamEstImgs = subSeamEstImgs;
        this.imgs = subImgs;

        let pano_ = [];
        const numImages = this.imgs.length;
        let corners = this.genMatrix(numImages);
        let imagesWarped = this.genMatrix(numImages);
        let masksWarped = this.genMatrix(numImages);
        let sizes = this.genMatrix(numImages);
        let masks = [];
        for (let i = 0; i < numImages; i++) {
            masks.push(this.genMatrix(this.seamEstImgs[i].length, this.seamEstImgs[i][0].length, 255)); 
        }

        let w = new Warper(this.warpedImgScale * this.seamWorkAspect);
        for (let i = 0; i < numImages; i++) {
            let K = this.cameras[i].K();
            K[0][0] *= this.seamWorkAspect;
            K[0][2] *= this.seamWorkAspect;
            K[1][1] *= this.seamWorkAspect;
            K[1][2] *= this.seamWorkAspect;

            const seamEstImgSize = {
                height: this.seamEstImgs[i].length,
                width: this.seamEstImgs[i][0].length
            };
            corners[i] = w.warp(seamEstImgSize, K, this.cameras[i].R, interpFlags, BORDER_REFLECT, imagesWarped[i]);
            sizes[i] = imagesWarped[i].size;
            const maskSize = {
                height: this.masks[i].length,
                width: this.masks[i][0].length
            };
            w.warp(maskSize, K, this.cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masksWarped[i]);
        }

        this.exposureComp.feed(conners, imagesWarped, masksWarped);
        for (let i = 0; i < numImages; i++) {
            this.exposureComp.apply(i, corners[i], imagesWarped[i], masksWarped[i]);
        }
        seamFinder.find(imagesWarped, corners, masksWarped);
        
        this.clear(this.seamEstImgs);
        this.clear(imagesWarped);
        this.clear(masks);
        

        let imgWarped = []; 
        let composeWorkAspect = 1;
        let isBlenderPrepared = false;
        let composeScale = 1;
        let isComposeScaleSet = false;
        let  camerasScaled = [];

        for (let idx = 0; idx < numImages; idx++) {
            let fullImg = this.imgs[idx];
            if (!isComposeScaleSet) {
                if (this.composeRes > 0) {
                    composeScale = Math.min(1.0, Math.sqrt(this.composeRes * 1e6 / fullImg.length * fullImg[0].length));
                }
                isComposeScaleSet = true;
                composeWorkAspect = composeScale / this.workScale;
                let warpScale = this.wapredImgScale * composeWorkAspect;
                w = new warp(warpScale);

                for (let i = 0; i < this.imgs.length; i++) {
                    camerasScaled[i].ppx *= composeWorkAspect;
                    camerasScaled[i].ppy *= composeWorkAspect;
                    camerasScaled[i].focal *= composeWorkAspect;

                    let sz = this.fullImgSizes[i];
                    if (Math.abs(composeScale - 1) > 0.1) {
                        sz.height = Math.round(this.fullImgSizes[i].height * composeScale);
                        sz.width = Math.round(this.fullImgSizes[i].width * composeScale);
                    }
                    let K = camerasScaled[i].K();
                    let roi = w.warpROI(sz, K, camerasScaled[i].R);
                    corners[i] = roi.tl();
                    sizes[i] = roi.size();
                }
            }
            if (Math.abs(composeScaled - 1) > 0.1) {
                this.resize(fullImg, img, composeScale, composeScale, INTER_LINEAR_EXACT);
            } else {
                img = fullImg;
            }
            this.clear(fullImg);
            let K = camerasScaled[idx].K();
            w.warp(img, K, this.cameras[idx].R, this.interpFlags, BORDER_REFLECT, imgWarped); 
            mask = this.genMatrix(img.length, img[0].length, 255);
            w.warp(mask, K, this.cameras[idx].R, INTER_NEAREST, BORDER_REFLECT, imgWarped); 
            this.clear(imgWarped);
            this.clear(mask);
            this.dilate(masksWarped[idx], dilatedMask);
            this.resize(dilatedMask, seamMask, maskWarped.length,  maskWarped[0].length); 
            this.bitwiseAnd(seamMask, maskWarped, maskWarped);
            if ( !isBlenderPrepared) {
                this.blender.prepare(cornes, sizes);
                isBlenderPrepared = true;
            }
            this.blender.feed(imgWarped, maskWarped, corners[idx]);
            blender.blend(pano, this.resultMask);
            return OK;
        }
    }

    stitch(images, masks, pano) {
        const status = this.estimateTransform(images, masks);
        if (!status) {
            return status;
        }
        return this.composePanorama(pano);
    }

    estimateTransform(images, masks) {
        this.imgs = images;
        this.masks = masks;
        let status = 0;
        if ((status = this.matchImages()) !== OK) {
            return status;
        }
        if ((status = this.estimateCameraParams()) !== OK) {
            return status;
        }
        return OK;
    }

    matchImages() {
        if (this.imgs.length < 2) {
            return ERR_NEED_MORE_IMGS;
        }

        const numImages = this.imgs.length;
        const height = this.imgs[0].length;
        const width = this.imgs[0][0].length;
        this.workScale = 1;
        this.seamWorkAspect = 1;
        this.seamScale = 1
        let isWorkScaleSet = false;
        let isSeamScaleSet = false;
        this.resize(this.features, numImages); 
        this.resize(this.seamEstImgs, numImages); 
        this.resize(this.fullImgSizes, numImages); 

        let featureFindImgs = []; 
        let featureFindMasks = []; 
        for (let i = 0; i < numImages; i++) {
            featureFindImgs.push(this.genMatrix(height, width));
            featureFindMasks.push(this.genMatrix(height, width));

            if (this.regRes < 0) {
                featureFindImgs[i] = this.imgs[i];
                this.workScale = 1;
                isWorkScaleSet = true;
            } else {
                if (!isWorkScaleSet) {
                    this.workScale = Math.min(1.0, Math.sqrt(this.regRes * 1e6 / (this.fullImgSizes[i].height * this.fullImgSizes[i].width)));
                    isWorkScaleSet = true;
                }
                this.resize(this.imgs[i], featureFindImgs[i], this.workScale, this.workScale, INTER_LINEAR_EXACT);
            }
            if (!isSeamScaleSet) {
                this.seamScale = Math.min(1.0, Math.sqrt(this.seamEstRes * 1e6 / (this.fullImgSizes[i].height * this.fullImgSizes[i].width)));
                this.seamworkAspect = this.seamScale / this.workScale;
                isWorkScaleSet = true;
            }
            
            if (this.masks) {
                this.resize(this.masks[i], featureFindMasks[i], this.workScale, this.workScale, INTER_LINEAR_EXACT);
            }
        }

        this.computeImgageFeatures(this.featuresFinder, featureFindImgs, this.features, featureFindMasks);
        this.clear(featureFindImgs);
        this.clear(featureFindMasks);

        this.featuresMatcher(this.features, this.pairMathces, this.matchingMask);
        this.featuresMatcher.collectGarbage();

        let detail = new Detail();
        this.indices = detail.leaveBiggestComponet(this.features, this.pairMatches, this.confThress);
        let subImgs = [];
        let subSeamEstImgs = [];
        let subFullImgSizes = [];
        for (let i = 0; i < this.indices.length; i++) {
            subImgs.push(this.imgs[this.indices[i]]);
            subSeamEstImgs.push(this.seamEstImgs[this.indices[i]]);
            subFullImgSizes.push(this.fullImgSizes[this.indices[i]]);
        }
        this.seamEstImgs = subSeamEstImgs;
        this.imgs = subImgs;
        this.fullImgSizes = subFullImgSizes;

        if (this.imgs.length < 2) {
            return ERR_NEED_MORE_IMGS;
        }
        return OK;
    }

    estimateCameraParams() {
        if (!this.estimator(this.features, this.pairMatches, this.cameras)) {
            return ERR_HOMOGRAPHY_EST_FAIL;
        }
        this.bundleAdjuster.setConfThresh(this.confThresh);
        if (!this.bundleAdjuster(this.features, this.pairMatches, this.cameras)) {
            return ERR_CAMERA_PARAMS_ADJUST_FAIL;
        }
        let focals = [];
        for (let i = 0; i < this.cameras.length; i++) {
            focals.push(this.cameras[i].focal);
        }
        focals.sort((a, b) => a - b);
        if (focals.length % 2 === 1) {
            this.warpedImageScale = focals[Math.floor(focals.length / 2)];
        } else {
            this.warpedImageScale = (focals[Math.floor(focals.length / 2 - 1)] + focals[Math.floor(focals.length / 2)]) * 0.5;
        }

        if (this.doWaveCorrect) {
            let rMats = [];
            for (let i = 0; i < this.cameras.length; i++) {
                rMats.push(this.cameras[i].R);
            }
            let detail = new Detail();
            detail.waveCorrect(rMats, this.waveCorrectKind);
            for (let i = 0; i < this.cameras.length; i++) {
                this.cameras[i].R = rMats[i];
            }
        }

        return OK;
    }
}

class CameraParams {
    constructor() {
        this.focal = 1;
        this.aspect = 1;
        this.ppx = 0;
        this.ppy = 0;
        this.R = this.genEye(3);
        this.T = this.genMatrix(3, 1, 0);
        this.projector = new PlaneProjector();
    }

    genMatrix(height, width, val) {
        if (height === undefined || height < 1 || width < 1) {
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

    K() {
        let K = this.genEye(3);
        k[0][0] = this.focal;
        k[0][2] = this.ppx;
        k[1][1] = this.focal * this.aspect;
        k[1][2] = this.ppy;
        return K;
    }
    update(options) {
        this.focal = options.focal || this.focal;
        this.aspect = options.aspect || this.aspect;
        this.ppx = options.ppx || this.ppx;
        this.ppy = options.ppy || this.ppy;
        this.R = options.R || this.R;
        this.T = options.T || this.T;
    }
}

class PlaneWarper {
    constructor() {
        this.projector = new PlaneProjector();
    }

    warpPoint(pt, K, R, T) {
        this.projector.setCameraParams(K, R, T);
        let uv = {
            x: 0,
            y: 0
        };
        this.projector.mapForward(pt,x pt.y, uv.x, uv.y);
    }

    buildMaps(srcSize, K, R, T, xmap, ymap) {
        this.projector.setCameraParams(K, R, T);
        // TL: Top Left
        let dstTL = {
            x: 0,
            y: 0
        };
        let dstBR = {
            x: 0,
            y: 0
        };
        this.detectResultROI(srcSize, dstTL, dstBR);
        // let dstSize = [dstBR.x - dstTL.x + 1, dstBR.y - dstTL.y + 1];
        let dstSize = {
            x: dstBR.x - dstTL.x + 1,
            y: dstBR.y - dstTL.y + 1
        }

        xmap = this.genMatrix(dstSize.y, dstSize.x, 0);
        ymap = this.genMatrix(dstSize.y, dstSize.x, 0);
        let x = 0;
        let y = 0;
        for (let v = dstTL.y; v <= dstBR.y; v++) {
            for (let u = dstTL.x; u <= dstBR.x; u++) {
                this.projector.mapBackward(u, v, x, y);
                xmap[v - dstTL.y][u - dstTL.x] = x;
                ymap[v - dstTL.y][u - dstTL.x] = y;
            }
        }
        const rect = {
            tl: {
                x: dstTL.x,
                y: dstTL.y
            },
            height: dstBR.y - dstTL.y,
            width: dstBR.x - dstTL.x
        };
        return rect; 
    }


    warp(src, K, R, T, interpMode, borderMode, dst) {
        let uxmap = [];
        let uymap = [];
        let dstROI = this.buildMaps(src.size, K, R, T, uxmap, uymap);
        dst = this.genMatrix(dstROI.height, dstROI.width, 0);
        this.remap(src, dst, uxmap, uymap, interpMode, borderMode);
        return dstROI.tl;
    }

    warpROI(srcSize, K, R, T) {
        this.projector.setCameraParams(K, R, T);
        let dstTL = {
            x: 0,
            y: 0
        };
        let dstBR = {
            x: 0,
            y: 0
        };
        this.detectResultROI(srcSize, dstTL, dstBR);
        const rect = {
            tl: {
                x: dstTL.x,
                y: dstTL.y
            },
            height: dstBR.y + 1 - dstTL.y,
            width: dstBR.x + 1 - dstTL.x
        };
        return rect; 
    }

    detectResultROI(srcSize, dstTL, dstBR) {
        let TLU = Number.MAX_VALUE;
        let TLV = Number.MAX_VALUE;
        let BRU = -Number.MAX_VALUE;
        let BRV = -Number.MAX_VALUE;
        let u = 0;
        let v = 0;
        this.projector.mapForward(0, 0, u, v);
        TLU = Math.min(TLU, u);
        TLV = Math.min(TLV, v);
        BRU = Math.max(BRU, u);
        BRV = Math.max(BRV, v);
        this.projector.mapForward(0, srcSize.height - 1, u, v);
        TLU = Math.min(TLU, u);
        TLV = Math.min(TLV, v);
        BRU = Math.max(BRU, u);
        BRV = Math.max(BRV, v);
        this.projector.mapForward(srcSize.width - 1, srcSize.height - 1, u, v);
        TLU = Math.min(TLU, u);
        TLV = Math.min(TLV, v);
        BRU = Math.max(BRU, u);
        BRV = Math.max(BRV, v);

        dstTL.x = TLU;
        dstTL.y = TLV;
        dstBR.x = BRU;
        dstBR.y = BRV;
    }
}

class PlaneProjector() {
    constructor() {
        this.rinv = [];
        this.rkinv = [];
        this.krinv = [];
        this.k = this.genArray(9, 0);
        this.t = this.genArray(3, 0);
        this.scale = 1.0;

    }

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

    mapForward(x, y, u, v) {
        rkinv = this.rkinv;
        scale = this.scale;
        t = this.t;
        let x_ = rkinv[0] * x + rkinv[1] * y + rkinv[2]; 
        let y_ = rkinv[3] * x + rkinv[4] * y + rkinv[5]; 
        let x_ = rkinv[6] * x + rkinv[7] * y + rkinv[8]; 
        x_ = t[0] + x_ / z_ * (1 - t[2]);
        y_ = t[1] + y_ / z_ * (1 - t[2]);

        u = scale * x_;
        v = scale * y_;


    }

    mapBackward(u, v, x, y) {
        krinv = this.krinv;
        scale = this.scale;
        t = this.t;
        u = u / scale - t[0];
        v = v / scale - t[1];
        let  z = 0;
        x = krinv[0] * u + krinv[1] * v + krinv[2] * (1 - t[2]); 
        y = krinv[3] * u + krinv[4] * v + krinv[5] * (1 - t[2]); 
        z = krinv[6] * u + krinv[7] * v + krinv[8] * (1 - t[2]); 

        x /= z;
        y /= z;
    }

    setCameraParams(K, R, T) {
        this.k = [K[0][0], K[0][1], K[0][2],
            K[1][0], K[1][1], K[1][2],
            K[2][0], K[2][1], K[2][2]];

        // let Rinv = matrix.transpose(R); 
        const R_ = new Matrix(R);
        const K_ = new Matrix(K);
        const Rinv = R_.transpose(); 
        const Kinv = inverse(K_);
        const RKinv = R_.mmul(Kinv); 
        const KRinv = K_.mmul(Rinv);

        this.rinv = [Rinv[0][0], Rinv[0][1], Rinv[0][2],
            Rinv[1][0], Rinv[1][1], Rinv[1][2],
            Rinv[2][0], Rinv[2][1], Rinv[2][2]];

        this.krinv = [KRinv[0][0], KRinv[0][1], KRinv[0][2],
            KRinv[1][0], KRinv[1][1], KRinv[1][2],
            KRinv[2][0], KRinv[2][1], KRinv[2][2]];
        
        this.t = [T[0][0], T[0][1], T[0][2]];
    }
}


class Detail {
    constructor() {
    }

    
}










































// export default SURF;

// test
const filePath1 = '../data/camera/c1.jpg'; 
const filePath2 = '../data/camera/c3.jpg'; 

const options = {
    hessianThreshold: 100,
    octaves: 4,
    layers: 2,
    extended: true,
    upright: false 
}

// console.log('Start ');
const surf1 = new SURF();
const surf2 = new SURF();

// console.log('surf1 ', surf);
const result1 = surf1.create(filePath1, options);
const result2 = surf2.create(filePath2, options);
// console.log('surf2 ', surf);
console.log('Result1 ', result1);
console.log('Result2 ', result2);
