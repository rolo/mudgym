// wasi_shim.js, the WASI preview1 host the MUD2 engine module needs in a browser, where wasmtime cannot run.
//
// The file is one expression, so both browser hosts evaluate the same text: MudGym's BrowserRuntime in Pyodide through
// run_js, and a page's own JavaScript through eval. It gives a function that instantiates the engine module, and whose
// unsupportedReason says first whether this browser can run the module at all. The engine carries its game data
// inside, so it asks the host only for clocks, randomness, short waits and somewhere to write stderr. Every other call
// answers the way wasmtime does for a guest given no directories, so the engine behaves as it does there.
(() => {
  // The engine needs WebAssembly exception handling with exnref, which browsers gained after everything else it uses.
  // An older browser rejects the engine with a compile error that names no browser, so hosts ask first whether this
  // module, (func (local exnref) (try_table)), validates.
  const EXNREF_PROBE = new Uint8Array([
    0, 97, 115, 109, 1, 0, 0, 0, 1, 4, 1, 96, 0, 0, 3, 2, 1, 0, 10, 10, 1, 8, 1, 1, 105, 31, 64, 0, 11, 11,
  ])
  const BROWSERS = "Chrome 137, Firefox 131, Safari 18.4 or later"

  function instantiateMudEngine(module) {
    const SUCCESS = 0
    const BAD_DESCRIPTOR = 8
    const INVALID = 28
    const CHARACTER_DEVICE = 2
    const ALL_RIGHTS = 0xffffffffffffffffn
    const CLOCK = 0
    const SUBSCRIPTION_BYTES = 48
    const EVENT_BYTES = 32
    // crypto.getRandomValues fills at most this many bytes at a time.
    const RANDOM_CHUNK = 65536

    let memory = null
    const view = () => new DataView(memory.buffer)
    const bytes = (pointer, length) => new Uint8Array(memory.buffer, pointer, length)
    const pendingText = { 1: "", 2: "" }
    const latin1 = new TextDecoder("latin1")

    const nanoseconds = (milliseconds) => BigInt(Math.round(milliseconds * 1e6))
    const standardStream = (descriptor) => descriptor >= 0 && descriptor <= 2

    const wasi = {
      clock_time_get(clock, _precision, out) {
        if (clock === 0) {
          view().setBigUint64(out, nanoseconds(performance.timeOrigin + performance.now()), true)
        } else if (clock === 1) {
          view().setBigUint64(out, nanoseconds(performance.now()), true)
        } else {
          return INVALID
        }
        return SUCCESS
      },
      random_get(pointer, length) {
        for (let offset = 0; offset < length; offset += RANDOM_CHUNK) {
          crypto.getRandomValues(bytes(pointer + offset, Math.min(RANDOM_CHUNK, length - offset)))
        }
        return SUCCESS
      },
      environ_sizes_get(count, size) {
        view().setUint32(count, 0, true)
        view().setUint32(size, 0, true)
        return SUCCESS
      },
      environ_get: () => SUCCESS,
      fd_fdstat_get(descriptor, out) {
        if (!standardStream(descriptor)) return BAD_DESCRIPTOR
        const stat = view()
        stat.setUint8(out, CHARACTER_DEVICE)
        stat.setUint16(out + 2, 0, true)
        stat.setBigUint64(out + 8, ALL_RIGHTS, true)
        stat.setBigUint64(out + 16, 0n, true)
        return SUCCESS
      },
      // stdout and stderr go to the console a line at a time, as wasmtime passes stderr through.
      fd_write(descriptor, vectors, count, written) {
        if (descriptor !== 1 && descriptor !== 2) return BAD_DESCRIPTOR
        const vector = view()
        let total = 0
        for (let index = 0; index < count; index += 1) {
          const pointer = vector.getUint32(vectors + index * 8, true)
          const length = vector.getUint32(vectors + index * 8 + 4, true)
          pendingText[descriptor] += latin1.decode(bytes(pointer, length).slice())
          total += length
        }
        const lines = pendingText[descriptor].split("\n")
        pendingText[descriptor] = lines.pop()
        for (const line of lines) (descriptor === 2 ? console.error : console.log)(`MUD2 engine: ${line}`)
        vector.setUint32(written, total, true)
        return SUCCESS
      },
      proc_exit(code) {
        throw new Error(`the MUD2 engine exited with code ${code}`)
      },
      // The engine waits only to pace what a player sees, as for the pause at a natural reset, and wasi-libc's sleep
      // retries for as long as a wait fails. A browser host must not block, so every clock wait ends at once and the
      // world carries on, and a wait on a descriptor names nothing.
      poll_oneoff(subscriptions, events, count, eventCount) {
        if (count === 0) return INVALID
        const poll = view()
        for (let index = 0; index < count; index += 1) {
          const subscription = subscriptions + index * SUBSCRIPTION_BYTES
          const event = events + index * EVENT_BYTES
          const type = poll.getUint8(subscription + 8)
          poll.setBigUint64(event, poll.getBigUint64(subscription, true), true)
          poll.setUint16(event + 8, type === CLOCK ? SUCCESS : BAD_DESCRIPTOR, true)
          poll.setUint8(event + 10, type)
          poll.setBigUint64(event + 16, 0n, true)
          poll.setUint16(event + 24, 0, true)
        }
        poll.setUint32(eventCount, count, true)
        return SUCCESS
      },
    }
    // No directory is preopened, so every descriptor past the standard streams, and every path, names nothing.
    for (const name of ["fd_advise", "fd_close", "fd_fdstat_set_flags", "fd_pread", "fd_prestat_get",
      "fd_prestat_dir_name", "fd_read", "fd_seek", "path_filestat_get", "path_link", "path_open", "path_readlink",
      "path_remove_directory", "path_rename", "path_unlink_file"]) {
      wasi[name] = () => BAD_DESCRIPTOR
    }

    for (const wanted of WebAssembly.Module.imports(module)) {
      if (wanted.module !== "wasi_snapshot_preview1" || !(wanted.name in wasi)) {
        throw new Error(`the MUD2 engine needs ${wanted.module}.${wanted.name}, which this host does not provide`)
      }
    }
    const instance = new WebAssembly.Instance(module, { wasi_snapshot_preview1: wasi })
    memory = instance.exports.memory
    return instance
  }

  // Why this browser cannot run the engine, in words for whoever is using it, or null when it can.
  instantiateMudEngine.unsupportedReason = () => {
    if (typeof WebAssembly !== "object") {
      return `This browser has no WebAssembly, which the MUD2 engine runs on. Try ${BROWSERS}.`
    }
    if (!WebAssembly.validate(EXNREF_PROBE)) {
      return `This browser is too old for the MUD2 engine, which needs WebAssembly exception handling. Try ${BROWSERS}.`
    }
    return null
  }
  return instantiateMudEngine
})()
