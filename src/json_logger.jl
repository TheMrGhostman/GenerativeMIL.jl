struct JSONLLogger
    io::IO
end

function JSONLLogger(path::String)
    io = open(path, "a")
    return JSONLLogger(io)
end

function log!(logger::JSONLLogger, log::NamedTuple)
    JSON3.write(logger.io, log)
    write(logger.io, '\n')
    flush(logger.io)
end

function log!(logger::JSONLLogger, log::NamedTuple, mode::String)
    log = merge(log, (;mode = mode))
    JSON3.write(logger.io, log)
    write(logger.io, '\n')
    flush(logger.io)
end

function close(logger::JSONLLogger)
    Base.close(logger.io)
end