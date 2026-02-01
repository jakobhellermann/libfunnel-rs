src/generated/libfunnel.rs: include/wrapper.h /usr/include/funnel/funnel.h /usr/include/funnel/funnel-vk.h /usr/include/funnel/funnel-egl.h /usr/include/funnel/funnel-gbm.h
	bindgen include/wrapper.h \
		--allowlist-function 'funnel_.*' \
		--allowlist-type 'funnel_.*' \
		--allowlist-var 'FUNNEL_.*' \
		--no-derive-debug \
		--no-layout-tests \
		--merge-extern-blocks \
		--rustified-enum 'funnel_.*' \
		| sponge src/bindings.rs
