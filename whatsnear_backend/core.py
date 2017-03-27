# -*- coding: utf-8 -*-


class Backend(object):
    def __init__(self):
        self._reader = None
        self._processor = None
        self._preprocessors = []
        self._writers = []

    def register_reader(self, reader, *args, **kwargs):
        self._reader = [reader, args, kwargs]

    def register_preprocessor(self, preprocessor, *args, **kwargs):
        self._preprocessors.append([preprocessor, args, kwargs])

    def register_processor(self, processor, *args, **kwargs):
        self._processor = [processor, args, kwargs]

    def register_writer(self, writer, *args, **kwargs):
        self._writers.append([writer, args, kwargs])

    def unregister_preprocessor(self, preprocessor):
        del self._preprocessors[preprocessor]

    def unregister_writer(self, writer):
        del self._writers[writer]

    def start(self):
        print('Start analyzing with %d preprocessors / %d writers' %
              (len(self._preprocessors), len(self._writers)))

        if self._reader is None or self._processor is None:
            print('Reader or Processor is required to run the backend.')
            return

        points = self._reader[0](*self._reader[1], **self._reader[2])

        for preprocessor in self._preprocessors:
            points = preprocessor[0](points, *preprocessor[1], **preprocessor[2])

        nodes, infos, edges = self._processor[0](points, *self._processor[1], **self._processor[2])

        for writer in self._writers:
            writer[0](nodes, infos, edges, *writer[1], **writer[2])
