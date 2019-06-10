/* 
// import SURF from './surf';
const SURF = require('./surf');

const filePath = '../data/camera/c1.jpg'; 

const options = {
    hessianThreshold: 100,
    octaves: 4,
    layers: 2,
    extended: 64,
    upright: true
}

console.log('Start ');
console.log('SURF', SURF);
const surf = new SURF();
const result = surf.create(filePath, options);
console.log('Result ', result);
*/
