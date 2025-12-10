// serial.js - WebSerial wrapper
export default function Serial({ onLine = ()=>{}, logger = console }) {
  let port = null;
  let reader = null;
  let writer = null;
  let keepReading = false;
  let textDecoder, inputDone, outputDone;

  async function connect() {
    port = await navigator.serial.requestPort();
    await port.open({ baudRate: 115200 });
    // writer
    const te = new TextEncoderStream();
    outputDone = te.readable.pipeTo(port.writable);
    writer = te.writable.getWriter();
    // reader
    const td = new TextDecoderStream();
    inputDone = port.readable.pipeTo(td.writable);
    reader = td.readable
      .pipeThrough(new TransformStream(new LineBreakTransformer()))
      .getReader();
    keepReading = true;
    readLoop().catch(e => logger.log('readLoop error:' + e));
  }

  async function disconnect() {
    keepReading = false;
    if (reader) { await reader.cancel(); reader = null; }
    if (inputDone) { await inputDone.catch(()=>{}); inputDone = null; }
    if (writer) { await writer.close(); writer = null; }
    if (outputDone) { await outputDone.catch(()=>{}); outputDone = null; }
    if (port) { await port.close(); port = null; }
  }

  async function readLoop() {
    while (keepReading && reader) {
      const { value, done } = await reader.read();
      if (done) break;
      if (value) onLine(value);
    }
  }

  async function send(line) {
    if (!writer) throw new Error('serial not open');
    await writer.write(line + '\n');
  }

  class LineBreakTransformer {
    constructor(){ this.container = ''; }
    transform(chunk, controller) {
      this.container += chunk;
      const lines = this.container.split(/\r?\n/);
      this.container = lines.pop();
      lines.forEach(l => controller.enqueue(l));
    }
    flush(controller) { if (this.container) controller.enqueue(this.container); }
  }

  return { connect, disconnect, send };
}
