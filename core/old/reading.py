import tensorflow as tf
fq = tf.train.string_input_producer(['features.csv', 'captions.csv'])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(fq)
# record_defaults = [[1] for i in range(100352+17)]
# col = tf.decode_csv(value, record_defaults=record_defaults)
# # features = tf.stack([col1, col2, col3, col4])
# min_after_dq = 1
# batch_size=64
# capacity = min_after_dq + 3 * batch_size
# cols = tf.train.shuffle_batch([col], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dq)
# features_batch,label_batch = cols[:100352,:],cols[100352:,:]
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
for i in range(1):
	# example, label = sess.run([features_batch, label_batch])
	col = sess.run(value)
	print col