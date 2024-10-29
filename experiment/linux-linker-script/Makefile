INCLUDE = include
ARCHS = riscv x86 loongarch arm64

CC = gcc
CFLAGS = -m64 -Og -g -Wall -Werror -I$(INCLUDE)

VERSION ?= 6.11
URL ?= https://raw.githubusercontent.com/torvalds/linux/refs/tags/v$(VERSION)

# arch header
RISCV_HEADERS = asm/pgtable.h asm/page.h asm/cache.h asm/thread_info.h asm/set_memory.h linux/sizes.h image-vars.h
X86_HEADERS = asm/asm-offsets.h asm/page_types.h asm/orc_lookup.h asm/vvar.h
ARM64_HEADERS = asm/hyp_image.h asm/kernel-pgtable.h asm/kexec.h asm/memory.h image.h

ARCH_HEADERS += $(RISCV_HEADERS) $(X86_HEADERS) $(ARM64_HEADERS)
ARCH_HEADERS_PATH = $(ARCH_HEADERS:%=$(INCLUDE)/%)

# linker header
LINKER_HEADERS += asm/vmlinux.lds.h asm-generic/vmlinux.lds.h asm-generic/codetag.lds.h
LINKER_HEADERS_PATH = $(LINKER_HEADERS:%=$(INCLUDE)/%)

# DIRS = $(INCLUDE)/asm $(INCLUDE)/asm-generic $(INCLUDE)/linux $(ARCHS:%=arch/%)
# $(info $(shell mkdir -p $(DIRS)))

all: $(ARCHS)

$(ARCHS): %: %.ld

%.ld: arch/%/vmlinux.lds.S $(LINKER_HEADERS_PATH) $(ARCH_HEADERS_PATH)
	$(CC) $(CFLAGS) -E $< -o $@

.PRECIOUS: arch/%/vmlinux.lds.S
arch/%/vmlinux.lds.S:
	@mkdir -p $(@D)
ifeq ($(strip $(LOCAL_REPO)),)
 	wget -q "$(URL)/$(@D)/kernel/$(@F)" -O $@
else
	@cp "$(LOCAL_REPO)/$(@D)/kernel/$(@F)" $@
endif

$(LINKER_HEADERS_PATH): %:
	@mkdir -p $(@D)
ifeq ($(strip $(LOCAL_REPO)),)
	wget -q "$(URL)/include/asm-generic/$(@F)" -O $@
else
	@cp "$(LOCAL_REPO)/include/asm-generic/$(@F)" $@
endif

%.h:
	@mkdir -p $(@D)
	@touch $@

clean:
	rm -rf $(INCLUDE) arch
	rm -f *.ld

.PHONY: all clean $(ARCHS)
