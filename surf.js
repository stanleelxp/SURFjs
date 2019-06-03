// import Jimp from 'jimp';
// import { solve } from 'ml-matrix'; // ~6.0.0
const Jimp = require('jimp');
const matrix = require('ml-matrix'); // ~6.0.0

const ORI_RADIUS = 6;
const ORI_WIN = 60;
const PATCH_SZ = 20;
const SURF_ORI_SEARCH_INC = 5;
const SURF_ORI_SIGMA = 2.5;
const SURF_DESC_SIGMA = 3.3;
const SURF_HAAR_SIZE0 = 9;
const SURF_HAAR_SIZE_INC = 6;

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

    // not in use
    genMatrices(N, height, width) {
        if (N === undefined || height === undefined || width === undefined || N < 1 || height < 1 || width < 1) {
            return [];
        } else {
            let arr1D = [];
            let arr2D = [];
            let arr3D = [];
            for (let i = 0; i < height; i++) {
                arr1D.push(0);
            }
            for (let i = 0; i < width; i++) {
                arr2D.push(arr1D.slice());
            }
            for (let i = 0; i < N; i++) {
                // wrong ?
                arr3D.push(arr2D.slice());
            }
            return arr3D;
        }
    }

    getMax2D(mat) {
        const height = mat.length;
        const width = mat[0].length;
        let max = mat[0][0];
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; j++) {
                if (max < mat[i][j]) {
                    max = mat[i][j];
                }
            }
        }
        return max;
    }

    createGaussianKernel(n, SIGMA) {
        let sum = 0;
        let kernel = [];
        const sigma = Math.max(SIGMA, 0);
        const scale = -0.5 / (sigma * sigma);
        const n2 = Math.floor(n / 2);
        for (let i = 0; i < n; i++) {
            const values = [];
            for (let j = 0; j < n; j++) {
                let d2 = (i - n2) * (i - n2) + (j - n2) * (j - n2);
                values.push(Math.exp(d2 * scale));
                sum += values[j];
            }
            kernel.push(values);
        }
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                kernel[i][j] /= sum;
            }
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
                    // console.log('idx ', idx);
                    sum[i][j] = sum[i][j - 1] + data[idx];
                    // continue;
                } else if (j === 0) {
                    const idx = i * width * 4;
                    // console.log('idx1 ', idx);
                    // console.log('i1 ', i);
                    sum[i][j] = sum[i - 1][j] + data[idx];
                } else {
                    const idx = i * width * 4 + j * 4;
                    // console.log('idx2 ', idx);
                    // FIXME can be faster
                    sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + data[idx];
                }
            }
        }
        return sum;
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
        // console.log('rH', dst.length);
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
        const invObject = matrix.solve(A, b);
        const x = invObject.data;
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
        const margin = Math.floor((sizes[layer + 1] / 2) / sampleStep + 1);
        // console.log('oct ', size);
        // console.log('dm ', maskSum);
        if (!maskSum) {
            this.resizeHaarPattern(dm, Dm, 9, size, maskSum[0].length);
        }

        // TODO
        // const step = Math.floor(dets[layer].step / dets[layer].elemSize());
        // const step = width;
        const step = dets[layer][0].length;
        console.log('step ', step);
        console.log('layer ', layer);
        console.log('margin', margin);

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

                     // TODO N9 max
                     // console.log('val ', val);
                     const maxInN9 = this.getMax2D(N9);
                     // console.log('mval ', maxInN9);
                     if(val > 0.9 * maxInN9) {
                         // console.log('val ', val);
                         // console.log('mval ', maxInN9);
                         const hCenter = hOffset + (size - 1) * 0.5;
                         const wCenter = wOffset + (size - 1) * 0.5;
                         // console.log('hoffset', hOffset);

                         // const kpt = this.genKeyPoint(hCenter, wCenter, sizes[layer], -1, val, octave, (traces[layer][j] > 0) - (traces[layer][j] < 0));
                         const kpt = {
                             x: wCenter,
                             y: hCenter, 
                             size: sizes[layer],
                             angle: -1,
                             response: val,
                             octave: octave,
                             classID: (traces[layer][j] > 0) - (traces[layer][j] < 0)
                         };
                         const ds = size - sizes[layer - 1];
                         const interpOK = this.interpolateKeypoint(N9, sampleStep, sampleStep, ds, kpt);

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
        this.keypoints.splice(0, this.keypoints.length);
        let index = 0;
        let middleIndex = 0;
        let step = sampleStep0;
        for (let octave = 0; octave < octaves; octave++) {
            for (let layer = 0; layer < octaveLayers + 2; layer++) {
                // TODO
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
    
    invoke(img, kps) {
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
        const patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        const _patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        const apt = this.genZeros(oriSampleBound);
        const aptw = this.genZeros(oriSampleBound);
        const DW = this.genZeros(PATCH_SZ * PATCH_SZ);
        const gOri = this.createGaussianKernel(2 * ORI_RADIUS + 1, SURF_ORI_SIGMA);
        const gDesc = this.createGaussianKernel(PATCH_SZ, SURF_DESC_SIGMA);
        const dSize = this.extended ? 128: 64;
        const height = this.height;
        const width = this.width;
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
                    const weight_ = gOri[i + ORI_RADIUS][0] * gOri[j + ORI_RADIUS][0];
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
                DW[i * PATCH_SZ + j] = gDesc[i][0] * gDesc[j][0];
            }
        }


        for (let k = 0; k < kps; k++) {
            const dx_t = this.genZeros(NX);
            const dy_t = this.genZeros(NY);
            const size = kp.size;
            const s = size * 1.2 / 9.0;
            const gradWaveSize = 2 * Math.round(2 * s);
            // const center = kp.pt;
            const center = {
                x: kp.x,
                y: kp.y
            };
            let kp = this.keypoints[k];
            let descriptorDir = 360.0 - 90.0;
            let squareMag = 0;
            let dOffset = 0;

            if (height < gradWaveSize || width < gradWaveSize) {
                kp.splice(0, kp.length); 
                continue;
            }
            if (!this.upright) {
                let a1 = 0;
                this.resizeHaarPttern(dx_s, dx_t, NX, 4, gradWaveSize, sum[0].length); 
                this.resizeHaarPttern(dy_s, dy_t, NX, 4, gradWaveSize, sum[0].length); 
                for (let k1 = 0; k1 < oriSamples; k1++) {
                    const x = Math.round(center.x + apt[k1].x * s - (gradWaveSize - 1) * 0.5);
                    const y = Math.round(center.y + apt[k1].y * s - (gradWaveSize - 1) * 0.5);
                    if (y < 0 || y >= height - gradWaveSize ||
                        x < 0 || x >= width - gradWaveSize) {
                        continue;
                    }
                    const sumOffset = {
                        hOffset: y,
                        wOffset: x 
                    };
                    const vx = this.calcHaarPattern(sum, dx_t, 2, sumOffset);
                    const vy = this.calcHaarPattern(sum, dy_t, 2, sumOffset);
                    X[a1] = vx * aptw[k1];
                    Y[a1] = vy * aptw[k1];
                    a1 += 1;
                }
                if (a1 === 0){
                    kp.splice(0, kp.length); 
                    continue;
                }
                //TODO phase
                // true mean in degrees
                // phase(X, Y, angle, true);
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
                    for (let j = 0; j < a1; j++) {
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
            kp.angle = descriptorDir;
            if (!this.descriptors || !this.descriptors[0]) {
                continue;
            }

            const winSize = Math.floor((PATCH_SZ + 1) * s);
            const win = this.genZeros(winSize, winSize);
            if (!this.upright) {
                descriptorDir *= Math.PI / 180.0;
                const sinDir = - Math.sin(descriptorDir);
                const cosDir = Math.cos(descriptorDir);
                const winOffset = -(win_size - 1) * 0.5;
                const startX = center.x + winOffset * cosDir + winOffset * sinDir;
                const starty = center.x - winOffset * cosDir + winOffset * sinDir;
                const width1 = width - 1;
                const height1 = height - 1;
                const data = img.bitmap.data;
                for (let i = 0; i < winSize; i++, startX += sinDir, startY += cosDir) {
                    const pixelX = startX;
                    const pixelY = startY;
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
                const startX = Math.round(center.x + winOffset);
                const startY = Math.round(center.y - winOffset);
                for (let i = 0; i < winSize; i++, startX++) {
                    const pixelX = startX;
                    const pixelY = startY;
                    for (let j = 0; j < winSize; j++, pixelY--) {
                        const x = Math.max(pixelX, 0);
                        const y = Math.max(pixelY, 0);
                        x = Math.min(x, width1);
                        y = Math.min(y, height1);
                        // win[i][j] =  data[y][x];
                        win[i][j] =  data[y * width * 4 + x];
                    }
                }
            }
            // TODO resize
            // not in use?
            // resize(win, _patch, _patch.length, 0, 0, INTER_AREA);
            // _patch = this.resize(win, _patch.length);
            for (let i =  0; i < PATCH_SZ; i++) {
                for (let j =  0; j < PATCH_SZ; j++) {
                    const dw = DW[i * PATCH_SZ + j];
                    const vx = (PATCH[i][j + 1] - PATCH[i][j] + PATCH[i + 1][j + 1] - PATCH[i + 1][j]) * dw;
                    vy = (PATCH[i + 1][j] - PATCH[i][j] + PATCH[i + 1][j + 1] - PATCH[i][j + 1]) * dw;
                    DX[i][j] = vx;
                    DY[i][j] = vy;
                }
            }
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
        if (!this.useProvidedKeyPoints) {
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
                    const { x, y } = this.keypoints[i];
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
            this.invoke(img, this.keypoints);
            let j = 0;
            for (let i = 0; i < kps; i++) {
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
                }
            }
        }
    }

    // create(hessianThreshold, octaves, layers, extended, upright) {
    create(filePath, options) {
        if (!filePath) {
            return;
        }
        // default: 100, 4, 2, 64, true
        this.filePath = filePath;
        this.hessianThreshold = options.hessianThreshold || 100;
        this.octaves = options.octaves || 4;
        this.layers = options.layers || 2;
        this.extended = options.extended || 64;
        this.upright = options.upright || true;

        this.detectAndCompute();
    }
};

// export default SURF;

// test
const filePath = '../data/camera/c1.jpg'; 

const options = {
    hessianThreshold: 100,
    octaves: 4,
    layers: 2,
    extended: 64,
    upright: true
}

// console.log('Start ');
const surf = new SURF();
// console.log('surf1 ', surf);
const result = surf.create(filePath, options);
// console.log('surf2 ', surf);
console.log('Result ', result);
