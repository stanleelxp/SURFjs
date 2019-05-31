import Jimp from 'jimp';

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
        this.sizes = [];
        this.dets = [];
        this.traces = [];
        this.descriptors = [];
        this.middleIndices = [];
        this.keypoints = [];
        /*
        this.keypoint = {
            x: 0,
            y: 0
            size: 0,
            angle: 0,
            octave: 0,
            response: 0,
            classID: -1 
        };
        */
        this.range = [0, 0];
        this.octaves = 4;
        this.octaveLayers = 3;
        this.sampleStep = 2;
        this.hessianThreshold = 100;
        this.useProvidedKeypoints = false;
        this.upright = false;
        this.extended = false;
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
            for (let i = 0; i < height; i++) {
                arr1D.push(0);
            }
            for (let i = 0; i < width; i++) {
                arr2D.push(arr1D);
            }
            return arr2D;
        }
    }

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
                arr2D.push(arr1D);
            }
            for (let i = 0; i < N; i++) {
                arr3D.push(arr2D);
            }
            return arr3D;
        }
    }

    calcHaarPattern(src, hf, N, hOffset, wOffset) {
        let d = 0;
        const width = src[0].length;
        const { hOffset, wOffset } = offset;
        if (width) {
            for (let k = 0; k < N; k++) {
                // d += (src[hf[k][0]] + src[hf[k][3]] - src[hf[k][1]] - src[hf[k][2]]) * hf[k][4];
                // TODO wOffset
                d += (src[hOffset + Math.floor(hf[k][0] / width)][(wOffset + hf[k][0]) % width] + 
                    src[hOffset + Math.floor(hf[k][3] / width)][(wOffset + hf[k][3]) % width] - 
                    src[hOffset + Math.floor(hf[k][1] / width)][(wOffset + hf[k][1]) % width] - 
                    src[hOffset + Math.floor(hf[k][2] / width)][(wOffset + hf[k][2]) % width]) * hf[k][4]; 
            }
        }
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
    }

    calcLayerDetAndTrace(sum, size, sampleStep, det, trace) {
        if (size > sum.length - 1 || size > sum[0].length - 1) {
            return;
        }

        const NX = 3;
        const NY = 3;
        const NXY = 4;
        const dx_s = [[0, 2, 3, 7, 1], [3, 2, 6, 7, -2], [6, 2, 9, 7, 1]];
        const dy_s = [[2, 0, 7, 3, 1], [2, 3, 7, 6, -2], [2, 6, 7, 9, 1]];
        const dxy_s = [[1, 1, 4, 4, 1], [5, 1, 8, 4, -1], [1, 5, 4, 8, -1], [5, 5, 8, 8, 1]];
        const height = sum.length;
        const width = sum[0].length;

        const Dx = this.genZeros(NX, 5); 
        const Dy = this.genZeros(NY, 5); 
        const Dxy = this.genZeros(NXY, 5); 

        this.resizeHaarPattern(dx_s, Dx, 9, size, width);
        this.resizeHaarPattern(dy_s, Dx, 9, size, width);
        this.resizeHaarPattern(dxy_s, Dxy, 9, size, width);

        const heightRange = 1 + (height - 1 - size) / sampleStep;
        const widthRange = 1 + (width - 1 - size) / sampleStep;
        const margin = Math.round(0.5 * size / sampleStep);
        // TODO
        for (let i = 0; i < heightRange; i++) {
            let sumIdx = i * sampleStep;
            let sumOffset = {
                hOffset: Math.floor(sumIdx / width),
                wOffset: sumIdx % width
            };
            for (let j = 0; j < widthRange; j++) {
                const dx = this.calcHaarPattern(sum, Dx, 3, sumOffset);
                const dy = this.calcHaarPattern(sum, Dy, 3, sumOffset);
                const dxy = this.calcHaarPattern(sum, Dxy, 4, sumOffset);
                sumIdx += sampleStep;
                sumOffset = {
                    hOffset: Math.floor(sumIdx / width),
                    wOffset: sumIdx % width
                };
                det[i + margin][j + margin] = dx * dy - 0.81 * dxy * dxy;
                trace[i + margin][j + margin] = dx + dy;
            }
        }
    }

    interpolateKeypoint(N9, dx, dy, ds, kpt) {
        const b = [-(N9[1][5] - N9[1][3]) * 0.5 , -(N9[1][7] - N9[1][1]) * 0.5, -(N9[2][4] - N9[0][4]) * 0.5];
        const A = [N9[1][3] - 2 * N9[1][4] + N9[1][5],
            (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25,
            (N9[2][5] - N9[2][3] - N9[0][5] + N9[0][3]) * 0.25,
            (N9[1][8] - N9[1][6] - N9[1][2] + N9[1][0]) * 0.25,
            N9[1][1] - 2 * N9[1][4] + N9[1][7],
            (N9[2][7] - N9[2][1] - N9[0][7] + N9[0][1]) * 0.25,
            (N9[2][5] - N9[2][3] - N9[0][5] + N9[0][3]) * 0.25,
            (N9[2][7] - N9[2][1] - N9[0][7] + N9[0][1]) * 0.25,
            N9[0][4] - 2 * N9[1][4] + N9[1][4]];

        // TODO
        const x = A.solve(b, decomp_lu);
        const isOK = (x[0] !== 0 || x[1] !== 0 || x2 !== 0) &&
            Math.abs(x[0]) <= 1 && Math.abs(x[1]) <= 1 && Math.abs(x[2]) <= 1;

        if (isOK) {
            kpt.pt.x += x[0] * dx;
            kpt.pt.y += x[1] * dy;
            kpt.size = Math.round(kpt.length + x[2] * ds);
        }
        return isOK;
    }

    buildInvoker(totalLayers) {
        for (let i = 0; i < totalLayers; i++) {
            this.calcLayerDetAndTrace();
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
        const height = this.sum.length;
        const width = this.sum[0].length;
        const layerHeight = height / sampleStep;
        const layerWidth = width / sampleStep;
        const hessianThreshold = this.hessianThreshold;
        const margin = (sizes[layer + 1] / 2) / sampleStep + 1;
        if (!maskSum[0]) {
            this.resizeHaarPattern(dm, Dm, 9, size, maskSum[0].length);
        }

        // TODO
        // const step = Math.floor(dets[layer].step / dets[layer].elemSize());
        // const step = width;
        const step = dets[layer][0].length;

        // for (let i = 0; i < layerHeight; i++) {
        //     for (let j = 0; j < layerWidth; j++) {
        for (let i = margin; i < layerHeight - margin; i++) {
             for (let j = margin; j < layerWidth - margin; j++) {
                const val = dets[layer][j];
                if (val > hessianThreshold) {
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
                    const N9 = [[d1[-step - 1], d1[-step], d1[-step + 1],
                        d1[-1], d1[0], d1[1],
                        d1[step - 1], d1[step], d1[step + 1]],
                        [d2[-step - 1], d2[-step], d2[-step + 1],
                            d2[-1], d2[0], d2[1],
                            d2[step - 1], d2[step], d2[step + 1]],
                        [d3[-step - 1], d3[-step], d3[-step + 1],
                            d3[-1], d3[0], d3[1],
                            d3[step - 1], d3[step], d3[step + 1]]];

                    if (!maskSum[0]) {
                    const maskSumOffset = {
                        hOffset: hOffset,
                        wOffset: wOffset 
                    };
                        const mval = this.calcHaarPattern(maskSum, Dm, 1, maskSumOffset);
                        if (mval < 0.5) {
                            continue;
                        }
                    }

                    // TODO N9 max
                    if(val > Math.max(N9)) {
                        const hCenter = hOffset + (size - 1) * 0.5;
                        const wCenter = wOffset + (size - 1) * 0.5;

                        // const kpt = this.genKeyPoint(hCenter, wCenter, sizes[layer], -1, val, octave, (traces[layer][j] > 0) - (traces[layer][j] < 0));
                        const kpt = {
                            x: wCenter,
                            y: hCenter, 
                            size: sizes[layer],
                            angle: -1,
                            response: val,
                            octave: octave
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
        const totalLayers = (octaveLayers + 2) * octaves;
        const middleLayers = octaveLayers * octaves;
        const octaves = this.octaves;
        const octaveLayers = this.octaveLayers;
        const height = this.sum.length;
        const width = this.sum[0].length;
        // this.dets = this.genMatrices(totalLayers, height, width);
        // this.traces = this.genMatrices(totalLayers, height, width);

        this.sizes = this.genZeros(totalLayers);
        this.sampleSteps = this.genZeros(totalLayers);
        this.middleIndices = this.genZeros(middleIndices);

        // clear()
        this.keypoints.splice(0, keypoint.length);
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
        keypoints.sort((a, b) => b.response - a.response);
    }
    
    createGaussianKernel(n, SIGMA) {
        let sum = 0;
        let kernel = [];
        const scale = -0.5 / (sigma * sigma);
        const sigma = Math.max(SIGMA, 0);
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

    // TODO
    invoker() {
        const oriSampleBound = (2 * ORI_RADIUS + 1) * (2 * ORI_RADIUS + 1);
        // TODO
        const apt = this.genZeros(oriSampleBound);
        const aptw = this.genZeros(oriSampleBound);
        const DW = this.genZeros(PATCH_SZ * PATCH_SZ);
        const gOri = createGaussianKernel(2 * ORI_RADIUS + 1, SURF_ORI_SIGMA);
        const gDesc = createGaussianKernel(PATCH_SZ, SURF_DESC_SIGMA);
        this.OriSamples = 0;
        for (let i = -ORI_RADIUS; i <= ORI_RADIUS; i++) {
            for (let j = -ORI_RADIUS; j <= ORI_RADIUS; j++) {
                if (i * i + j * j <= ORI_RADIUS * ORI_RADIUS) {
                    const weight_ = gOri[i + ORI_RADIUS][0] * gOri[j + ORI_RADIUS][0];
                    apt.push([i, j]);
                    aptw.push(weight_);
                    this.OriSamples += 1;
                }
            }
        }
        for (let i = 0; i < PATCH_SZ; i++) {
            for (let j = 0; j < PATCH_SZ; j++) {
                DW[i * PATCH_SZ + j] = gDesc[i][0] * gDesc[j][0];
            }
        }
    }

    invoke(N) {
        const NX = 2;
        const NY = 2;
        const dx_s =[[0, 0, 2, 4, -1], [2, 0, 4, 4, 1]];
        const dy_s =[[0, 0, 4, 2, 1], [0, 2, 4, 4, -1]];
        const oriSampleBound = (2 * ORI_RADIUS + 1) * (2 * ORI_RADIUS + 1);
        const X = [];
        const Y = [];
        const angle = [];
        const patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        const _patch = this.genZeros(PATCH_SZ + 1, PATCH_SZ + 1);
        const DX = [];
        const DY = [];
        const dsize = this.extended ? 128: 64;
        // TODO
        const start = 0;
        const end = N;
        const height = sum.length;
        const width = sum[0].length;

        for (let k = k1; k < k2; k++) {
            let vec = [];
            let dx_t = [];
            let dy_t = [];
            let kp = {};
            let size = kp.length;
            let s = size * 1.2 / 9.0;
            // const center = kp.pt;
            const center = {
                x: kp.x,
                y: kp.y
            };

            let gradWaveSize = 2 * Math.round(2 * s);
            if (height < gradWaveSize || width < gradWaveSize) {
                kp.splice(0, kp.length); 
                continue;
            }
            let descriptorDir = 270.0;
            if (this.upright === 0) {
                this.resizeHaarPttern(dx_s, dx_t, NX, 4, gradWaveSize, sum[0].length); 
                this.resizeHaarPttern(dy_s, dy_t, NX, 4, gradWaveSize, sum[0].length); 
                for (let k1 = 0, let nangle = 0; k1 < oriSamples; k1++) {
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
                    X[nangle] = vx * aptw[k1];
                    Y[nangle] = vy * aptw[k1];
                    nangle += 1;
                }
                if (nangle === 0){
                    kp.splice(0, kp.length); 
                    continue;
                }
                //TODO phase
                phase(X, Y, angle, true);
                let bestX = 0;
                let bestY = 0;
                let descriptorMod = 0;
                for (let i = 0; i < 360; i++) {
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
            kp.angle = descriptorDir;
            if (!descriptor || !descriptor.data) {
                continue;
            }

            const winSize = Math.round((PATCH_SZ + 1) * s);
            const win = this.genZeros(winSize, winSize);
            if (!upright) {
                descriptorDir *= Math.PI / 180.0;
                const sinDir = - Math.sin(descriptorDir);
                const cosDir = Math.cos(descriptorDir);
                const winOffset = -(win_size - 1) * 0.:
                const startX = center.x + winOffset * cosDir + winOffset * sinDir;
                const starty = center.x - winOffset * cosDir + winOffset * sinDir;
                const width1 = img.bitmap.width - 1;
                const height1 = img.bitmap.height - 1;
                // TODO step
                const imgStep = img.step;
                const data = img.bitmap.data;
                for (let i = 0; i < winSize; i++, startX += sinDir, startY += cosDir) {
                    const pixelX = startX;
                    const pixelY = startY;
                    for (let j = 0; j < winSize; j++, pixelX += cosDir, pixelY -= sinDir) {
                        const ix = Math.floor(pixelX);
                        const iy = Math.floor(pixelY);
                        if (ix < width1 && iy < height1) {
                            const a = pixelX - ix;
                            const b = pixelY - iy;
                            win[i][j] = Math.round(data[0] * (1.0 - a) * (1.0 - b) + data[1] * a * (1.0 - b) + 
                                data[imgStep] * (1.0 - a) * b + data[imgStep + 1] * a * (1.0 - b));
                        } else {
                            const x = Math.min(Math.max(Math.round(pixelX), 0), width1);
                            const y = Math.min(Math.max(Math.round(pixelY), 0), width1);
                            win[i][j] = data[y][x];
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
                            win[i][j] =  data[y][x];
                        }
                    }
                }
                // TODO resize
                resize(win, patch, patch.length, 0, 0, INTER_AREA);
                for (let i =  0; i < PATCH_SZ; i++) {
                    for (let j =  0; j < PATCH_SZ; j++) {
                        const dw = DW[i * PATCH_SZ + j];
                        cosnt vx = (PATCH[i][j + 1] - PATCH[i][j] + PATCH[i + 1][j + 1]  PATCH[i + 1][j]) * dw; 
                        cosnt vx = (PATCH[i + 1][j] - PATCH[i][j] + PATCH[i + 1][j + 1]  PATCH[i][j + 1]) * dw; 
                        DX[i][j] = vx;
                        DX[i][j] = vy;
                    }
                    vec = descriptors[k];
                    for (let k1 = 0; k1 < dsize; k1++) {
                        vec[k1] = 0;
                    }
                    let squareMag = 0;
                    if (this.extended) {
                        for (let i = 0; i < 4; i++) {
                            for (let j = 0; j < 4; j++) {
                                for (let y = i * 5; y < i * 5 + 5; y++) {
                                    for (let x = j * 5; x < j * 5 + 5; x++) {
                                        const tx = DX[y][x];
                                        const ty = DY[y][x];
                                        // TODO fabs
                                        if (ty >= 0) {
                                            vec[0] += tx;
                                            vec[1] += Math.abs(tx);
                                        } else {
                                            vec[2] += tx;
                                            vec[3] += Math.abs(tx);
                                        }
                                        if (tx >= 0) {
                                            vec[4] += ty;
                                            vec[5] += Math.abs(ty);
                                        } else {
                                            vec[6] += ty;
                                            vec[7] += Math.abs(ty);
                                        }
                                    }
                                }
                                for (let k1 = 0; k1 < 8; k1++) {
                                    squareMag += vec[k1] * vec[k1];
                                }
                                // TODO
                                vec += 8;
                            }
                        }
                    } else {
                        for (let i = 0; i < 4; i++) {
                            for (let j = 0; j < 4; j++) {
                                for (let y = i * 5; y < i * 5 + 5; y++) {
                                    for (let x = j * 5; x < j * 5 + 5; x++) {
                                        const tx = DX[y][x];
                                        const ty = DY[y][x];
                                        vec[0] += tx;
                                        vec[1] += ty;
                                        vec[2] += Math.abs(tx);
                                        vec[3] += Math.abs(ty);
                                    }
                                }
                                for (let k1 = 0; k1 < 4; k1++) {
                                    squareMag += vec[k1] * vec[k1];
                                }
                                // TODO
                                vec += 4;
                            }
                        }
                    }
                    vec = descriptors[k];
                    const scale = 1.0 / (Math.sqrt(squareMag + 1e-6));
                    for (let k1 = 0; k1 < 4; k1++) {
                        vec[k1] *= scale;
                    }
                }
            }
        }
    }
    
    matrixScalarCompare(mat, scalar, type) {
        const height = mat.length;
        const width = mat[0].length;
        const res = this.genZeros(height, width);
        for (let i = 0; i < height; i++) {
            for (let i = 0; i < height; i++) {
                const val = mat[i][j];
                if (type === 'min') {
                    res[i][j] = val < scalar ? val : scalar;
                } else if (type === 'max') {
                    res[i][j] = val > scalar ? val : scalar;
                } else {
                    res[i][j] = val;
                }
            }
        }
        return res;
    }

    detectAndCompute(mask, keypoints, descriptors, useProvidedKeypoints) {
        const img = Jimp.read('/c1.jpg');
        img = img.greyscale();
        this.sum = this.integral(img);
        if (!this.useProvidedKeyPoints) {
            if (!this.mask) {
                const mask1 = this.matrixScalarCompare(mask, 1, 'min');
                this.maskSum = this.integral(mask1);
            }
            this.fastHessianDetector();
            if (!this.mask) {
                for (let i = 0; i < keypoints.length;) {
                    const pt = keypoints[i].pt;
                    if (this.mask[pt.y][pt.x] === 0) {
                        // TODO
                        // erase
                        keypoints.splice(i, 1);
                        continue; // keep "i"
                    }
                    i++;
                }
            }
        }
        const N = keypoints.length;
        if (N > 0) {
            const dWidth = this.extended ? 128: 64;
            if (doDescriptors) {
                this.descriptors = this.genZeros(N, dWidth);
            }
            this.invoke(N);

            for (let i = 0, let j = 0; i < N; i++) {
                if (kepoints[i].length > 0) {
                    if (i > j) {
                        keypoints[j] = keypoints[i];
                        if (doDescriptors) {
                            // TODO
                            // memcpy(descriptors.ptr(j), descriptors.ptr(i), dsize);
                        }
                    }
                    j += 1;
                }
            }
            if (N > j) {
                // TODO
                keypoints.resize(N);
                if (doDescriptors) {
                    const d = [];
                    for (let k = 0; k < N; k++) {
                        d.push(descriptors[k]);
                    }
                    this.descriptors = d;

                }
            }
        }
    }

    integral(img) {
        const height = img.bitmap.height;
        const width = img.bitmap.width;
        const data = img.bitmap.data;
        const sum = this.genZeros(height, width);
        // TODO
        for (let i = 0; i < height; i++) {
            for (let j = 0; j < width; i++) {
                if (i === 0) {
                    const idx = j * 4;
                    sum[0][j] += data[idx];
                    continue;
                } else if (j === 0) {
                    const idx = i * width * 4;
                    sum[i][0] += data[idx];
                } else {
                    const idx = i * width * 4 + j * 4;
                    sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + data[idx];
                }
            }
        }

        return sum;
    }

    create(threshold, octaves, layers, extended, upright) {
        this.threshold = threshold;
        this.octaves = octaves;
        this.layers = layers;
        this.extended = extended;
        this.upright = upright;

        this.detectAndCompute();
    }

}
